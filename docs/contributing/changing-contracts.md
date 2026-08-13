# Changing Contracts

> **Status:** Active | **Audience:** Core contributors | **Authority:** Operational | **Applies to:** Aptus 0.2 | **Owner:** Architecture | **Last reviewed:** 2026-08-05 | **Review by:** 2026-10-27

Aptus contracts bind facts, decisions, generated files, runtime evidence, and
completion. A field change can alter identity even when its JSON shape looks
compatible. Decide the semantic effect before editing code.

## Contract inventory

| Contract | Current identifier | Primary authority |
|---|---|---|
| Facts | `aptus.facts.v3` | `domain.py` and interface request models |
| Training plan | `aptus.training-plan.v6` | `domain.py` and `plan_contract.py` |
| Model-policy snapshot | `aptus.model-policy-snapshot.v1` | `model_compatibility.py` and `policy_snapshot.py` |
| Model compatibility decision | `aptus.model-compatibility.v2` | `domain.py`, `model_compatibility.py`, and `plan_contract.py` |
| Model inspection receipt | `aptus.model-inspection-receipt.v1` | `inspection.py`, `model_compatibility.py`, and interface response models |
| Model policy binding | `aptus.model-policy-binding.v1` | `model_compatibility.py`, `planning.py`, and `plan_contract.py` |
| Memory formula | `aptus-memory-v2` | `planning.py` |
| MLX memory formula | `aptus-memory-mlx-v2` | `planning.py` and the method registry |
| Method descriptor | `aptus.method-descriptor.v1` | `methods/contracts.py` and registry |
| Bundle manifest | `aptus.bundle.v3` | `generation.py` and `plan_contract.py` |
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
| CUDA reload evidence | `aptus.cuda-reload-evidence.v1` | measured M7-C fresh-process PEFT reload; not a CUDA parent gate |
| MLX artifact manifest | `aptus.mlx-artifact-manifest.v1` | generated MLX action owner and parent verifier |
| Model architecture contract | `aptus.model-architecture-contract.v1` | `plan_contract.py` and generated MLX runtime |
| MLX model load binding | `aptus.mlx-model-load-binding.v3` | generated MLX runtime and parent verifier |
| MLX model parameter census | `aptus.mlx-model-parameter-census.v1` | generated MLX runtime and parent verifier |
| MLX packed checkpoint | `aptus.mlx-packed-checkpoint.v1` | generated MLX runtime and parent verifier |
| MLX unified-memory admission | `aptus.mlx-unified-memory-admission.v2` | generated MLX runtime and parent verifier |
| MLX model-data evidence | `aptus.mlx-model-data-evidence.v1` | generated MLX validator and parent verifier |
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
memory, status, and resource facts. Plan identity binds the schema and formula
versions, normalized facts, the ordered candidate IDs, the recommendation, the
semantic policy decision and source, `model_policy_snapshot_sha256`, the
optional inspection receipt with its nested explanatory decision reason
excluded, and canonical evidence records. Bundle identity binds the plan digest and
compiler-managed file manifest.

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

The bundle copies `plan_contract.py`, `policy_snapshot.py`, and
`runtime_lease.py`. It emits trainer, runner, preflight, validator, and MLX
reload programs from package resources under `src/aptus/_bundle_programs/`.
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

Package-free portable validation evaluates the bundle's frozen snapshot. It
must prove snapshot contract, canonical encoding, digest bindings, and decision
parity, but it has no installed host or current registry and cannot determine
host policy currency. Installed Aptus owns that separate currency boundary. It
must compare the bound snapshot digest and decision with the current registry
during host-managed submission, pilot authorization, worker launch, and the
completion verification and promotion transaction. A current-registry mismatch
requires replanning; portable integrity success does not waive that result.

## Persisted state and compatibility

Current plan and bundle validators require exact schema identifiers. There is no
general artifact migration command. Never reinterpret an old artifact under new
semantics without an explicit reader and migration policy.

The current plan reader accepts only `aptus.training-plan.v6`. A saved v4, v3,
or v2 plan, or a plan with no schema identifier, stays byte-for-byte preserved but
enters `replan_required`. A v5 plan also requires replanning when its decision,
snapshot digest, policy version, or registered path differs from the current
registry. The CLI
cannot compile it. The API cannot rehydrate, compile, or recover it, and project
recovery does not append a revision. Bootstrap omits the old plan from the
executable workspace and returns the source identity plus the required schema.
The operator must create a deterministic v5 plan from the preserved facts.
Changing only `schema_version` is not migration because v5 binds the policy
snapshot digest in addition to the decision, provenance source, receipt,
candidate links, and exact path binding.

For an incompatible change, choose one of these:

- keep a versioned reader and prove its old semantics;
- reject the older version with a clear finding;
- provide an explicit no-clobber migration that records source and destination
  identities.

Do not mutate a persisted plan, bundle, validation report, job, or run in place
to make it appear current.

## Model-policy provenance changes

Policy IDs and path IDs are stable public identities. Change the semantic
policy version when an existing policy's predicates, required provenance,
reason codes, evidence IDs, or emitted paths change. Introduce a new path ID
when its method, placement, runtime contract, adapter profile, target modules,
required gates, or evidence changes.

Recompute and validate both digests independently. `subject_facts_sha256`
covers only compatibility inputs. `observed_facts_sha256` covers every
provider-declared or inferred planning fact carried into a receipt. Parameters
and training permission remain user-attested and must never enter that receipt.
Every candidate links to the decision, while only an exact path match carries a
binding. Content hashes are tamper-evident, not authenticated signatures.
Receipt entries are limited to `provider-declared` and `inferred`, cover every
non-null compatibility subject field, and include at least one
provider-declared subject observation.

Evidence records are code-owned semantic inputs. Changing a record's claim,
source, source kind, scope, confidence, or revision requires a new evidence ID,
updated policy and method references, and a regenerated plan identity. The
portable validator must continue to require exact canonical record content and
the exact union cited by candidates.

Adding a policy is a registry-data change even when the snapshot schema remains
compatible. Supply portable claims, exact constraints, reason keys and codes,
paths, adapter profile, required provenance fields, and canonical evidence; do
not add a host callback or duplicate family branch. Claims may intentionally
use only the discriminating identity fields, while the exact identity
constraint still binds family, model type, and architecture. Prove that nearby
dense and sparse identities are not captured. Required receipt provenance must
come from the matched policy in the loaded snapshot rather than from another
policy's hard-coded field set.

Every added policy changes canonical snapshot bytes and their SHA-256. Replan
even when an older plan's subject would receive the same semantic decision, then
cross-bind the new digest through the v5 plan, v3 manifest, manifested snapshot,
installed-host currency checks, and package-free evaluator. Keep implementation
evidence distinct from runtime evidence. A historical measured record may
support a policy decision only within its exact artifact, revision, host,
runtime, dataset, and contract scope; it cannot establish a current run or a
broader artifact claim.

Phase 4 introduced a deterministic `aptus.model-policy-snapshot.v1`, its generic
portable evaluator, `aptus.training-plan.v6`, and `aptus.bundle.v3`. Generate
the snapshot twice and require byte-for-byte identity. Its SHA-256 must agree
across the plan, manifest, and manifested snapshot file. Host-versus-portable
decision parity is a required contract test. Also test package-free frozen
snapshot integrity independently from installed-host current-registry currency;
the former cannot establish the latter.

Phase 5 completed removal of browser-side policy reconstruction. The maintained
client now strictly decodes the server v2 decision, nested paths, optional
receipt, and each candidate's nullable binding, then presents artifact match,
selected candidate path, and evidence readiness separately. Report-backed
presentation requires exact plan, candidate, and model-revision bindings and
separates validation completeness from launch admission. Exact path equality
requires a non-null binding; unbound or rejected candidates receive no
synthesized policy ladder or action. The typed HTTP 422 `no_feasible_plan`
response preserves the same policy chain as successful planning, correlates it
with the request and receipt, requires complete rejected candidate tuples, and
remains non-compilable. Phase 6 has since implemented the second registry-driven
`model.qwen2-24l.mlx-qlora` configuration-footprint policy and
`mlx-lm.qlora.single.dense-causal-lm.v1` path. The
[2026-08-05 Qwen2 MLX-LM exact-source refresh](../operations/evidence/2026-08-05-qwen2-mlx-lm-exact-source-refresh/README.md)
supplies current-contract v5/v3 runtime evidence at its exact acceptance source
with two fresh, clean
`measured-run-pass` repetitions for the exact pinned artifact, source commit
`719255153e3fc7e38e83b5ff826d587e5e58bf80`, source tree, Apple M5 Pro host,
Python/MLX runtime, dataset, policy snapshot, and bundle fingerprint
`ca2548cf8469fb9867f1558428803b1c9f7c19f48cba754fdb602643f23d1919`.
Relative to the unchanged [original acceptance
baseline](../operations/evidence/2026-08-05-qwen2-mlx-lm-acceptance/README.md),
only manifested operator `README.md` and `runbook.md` changed; runtime programs
and requirements remained byte-identical. The policy
remains a configuration footprint, not an artifact allowlist; other matching
artifacts still require their own gates, and the result does not qualify CUDA
or establish safety, model quality, performance, production throughput,
production readiness, or release readiness. `aptus.api.v1`,
`aptus.facts.v3`,
`aptus.model-policy-snapshot.v1`, and `aptus.runtime-contract.v1` remain
unchanged; the acceptance closeout does not change the already-bound snapshot.
The registry row, emitted path, and compiler make the reviewed configuration
eligible for that conditional execution path. They do not transfer either the
MLX acceptance or the separate [exact CUDA
acceptance](../operations/evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/README.md)
to another artifact, bundle, host, or runtime.

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

For server-owned model policy, runtime normalization must reject extra or
missing keys, unsupported versions or closed vocabulary, malformed identities,
and any disagreement across decision, path, receipt, candidate, and binding.
Success and typed `no_feasible_plan` responses use the same cross-record rules.
Require both responses to carry a model subject matching the submitted model ID
and immutable revision, then correlate the expected source and receipt identity.
Require each candidate's method, distribution, status, feasibility, rejection
reasons, targets, runtime contract, decision link, and binding. No-feasible rows
must all be rejected. In a successful plan, decode the recommendation and its
listed row independently, then require full structural equality across the
complete candidate records, not equality of only a selected execution tuple.
Object key order is irrelevant; array order remains part of the contract.

A provider path-matched receipt must name a satisfied `provider-declared`
provenance requirement and contain provider-declared evidence. Inferred-only
provenance cannot satisfy the flag even when every other receipt identity agrees.

Keep a candidate's nullable binding distinct from its runtime contract, but
reject null when that complete tuple equals an emitted policy path. Do not infer
policy validation levels from a runtime contract for unbound or rejected rows.
Use a validation report only when its `bindings.plan_id`,
`bindings.candidate_id`, and `bindings.model_revision` match the current
selection. Reuse that exact predicate for model-policy evidence, workflow-stage
completion, and validation or run action enablement.

Model validation evidence as incomplete or complete. Treat launch admission as
an optional typed tuple: `authorization_status` is exactly `current`,
`deferred`, or `blocked`; current requires `authorization_current: true` and no
`authorization_error`; deferred or blocked requires false and a non-empty
diagnostic. If the tuple has no non-null member, admission is not checked.
Reject partial or contradictory claims. Never branch on diagnostic prose, and do not fabricate a
new report or authorization state after a generic training-request failure;
surface the request error while preserving the prior report. A non-current
status is not by itself stale policy or a replan instruction; only the separate
`replan_required` lifecycle result carries that meaning.

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
- [ ] Evidence content and evidence IDs change together and remain canonical.
- [ ] Unknown, malformed, stale, and tampered forms fail closed.
- [ ] Stale classification validates historical coherence without applying a
      newer mutable target catalog to the old plan.
- [ ] Malformed JSON scalar types produce typed validation errors, not process
      exceptions.
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
