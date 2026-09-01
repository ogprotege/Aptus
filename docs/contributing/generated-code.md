# Generated Code and Bundle Changes

> **Status:** Active | **Audience:** Compiler and runtime contributors | **Authority:** Operational | **Applies to:** Aptus 0.2 | **Owner:** Artifact compiler | **Last reviewed:** 2026-08-05 | **Review by:** 2026-10-27

A compiled bundle is a portable product artifact. Its Python programs,
configuration, data copies, reports, and manifest must agree with the selected
plan. Change the generator source, never a compiled bundle in place.

## Generated sources

Canonical runtime programs are package resources under
[`src/aptus/_bundle_programs/`](../../src/aptus/_bundle_programs/).
[`src/aptus/generation.py`](../../src/aptus/generation.py) selects and emits
those exact bytes:

| Resource | Bundle file | Responsibility |
|---|---|---|
| `_bundle_programs/cuda/train.py` | `train.py` | Model/data preparation, pilot and full training, census, splitting, checkpoints, metrics, and export |
| `_bundle_programs/cuda/run.py` | `run.py` | Portable full-run parent, launch, recovery, aggregate exit, artifact verification, and promotion |
| `_bundle_programs/cuda/preflight.py` | `preflight.py` | Selected-method synthetic CUDA work and measured census |
| `_bundle_programs/cuda/validate.py` | `validate.py` | Portable validation ladder and two-phase pilot orchestration |
| `_bundle_programs/mlx/train.py` | `train.py` | Exact-target MLX adapter updates for smoke, pilot, and full actions |
| `_bundle_programs/mlx/run.py` | `run.py` | Owned uninterrupted MLX action outputs, artifact sealing, and full-run export |
| `_bundle_programs/mlx/reload.py` | `reload.py` | Fresh-process adapter reload and one-to-four-token generation |
| `_bundle_programs/mlx/preflight.py` | `preflight.py` | Apple-silicon platform and pinned `mlx`/`mlx-lm` version gate, spawned by `validate.py`; takes no arguments and owns no level ladder |
| `_bundle_programs/mlx/validate.py` | `validate.py` | MLX validation ladder, attestations, and fail-closed evidence checks |

The compiler also copies current package sources into:

- `plan_contract.py` for identity and manifest checks;
- `policy_snapshot.py` for exact snapshot validation and generic policy
  evaluation; and
- `runtime_lease.py` for portable host-global coordination.

These programs must run from the bundle environment without importing the Aptus
application package.

## Other compiler-owned outputs

The same module emits:

- portable `plan.json` and fact profiles;
- canonical `policy/model-policy-snapshot.v1.json` generated from the current
  host model-policy registry;
- `candidates.json`, `decision-report.md`, and `evidence.jsonl`;
- exact direct pins in `requirements.txt`;
- `config/trainer.json` and Accelerate configuration;
- MLX-LM configuration, disjoint padded MLX data files, and split contract when
  that runtime is selected;
- copied source data, canonical training JSONL, and pilot pressure rows;
- bundle README and runbook;
- `bundle-manifest.json`;
- an atomic directory and deterministic no-clobber ZIP.

The expected tree is defined in the
[bundle-manifest reference](../reference/bundle-manifest.md). Runtime reports
and output directories are mutable exclusions with separate evidence contracts.

## Safe change workflow

1. Identify the owning contract, generated consumer, and host verifier.
2. Add a failing test for the intended behavior and at least one invalid form.
3. Edit the packaged program resource or compiler helper.
4. Update copied/shared contract code when needed.
5. Compile a fresh fixture into an absent or empty path.
6. Run contract and static validation.
7. Diff the complete generated tree, not only the edited script.
8. Import or execute generated modules through isolated test fixtures.
9. For policy-affecting changes, prove exact host/portable decision parity,
   malformed-snapshot rejection, frozen-snapshot standalone behavior, and
   installed-host currency rejection.
10. Run the affected target-runtime gate. Use the CUDA pilot for CUDA resources
   and the MLX-LM evidence ladder for MLX resources.
11. Update bundle, validation, capability, and operations documentation.

Do not reuse an old output directory. No-clobber behavior is part of the
compiler contract and its safety tests.

## Preserve self-containment

Generated programs can use the direct pinned training stack and Python standard
library. They must not depend on an unlisted repository path, developer virtual
environment, current working directory outside the bundle, shell alias, or
globally installed Aptus package.

Use paths relative to the resolved bundle root. Reject symlinks and unsafe
relative paths where the contract requires it. Use the active Python
interpreter's module invocation for Accelerate rather than a shell-resolved
executable.

Package-free entrypoints must evaluate the canonical snapshot embedded in the
bundle. They must not import the installed Aptus policy registry or claim that
the frozen snapshot is still current. Installed Aptus owns the separate current
registry check used by host admission and managed execution.

The host currently serializes five ordered registry rows into that snapshot:
Qwen3 MoE attention-only QLoRA, dense 24-layer Qwen2 QLoRA, dense Gemma 4,
Gemma 4 unified, and Gemma 4 MoE attention-only QLoRA. Snapshot
generation and portable evaluation must remain registry-driven; generated code
must not select a row through a policy-ID singleton branch. A policy's
`any_identity` claims may cover only the exact-identity fields that distinguish
that row, while its `exact_identity` constraint still binds family, model type,
and architecture. Provider-inspection validation must read
`required_provenance_fields` from the matched policy rather than apply one
model's field set globally.

## Preserve identity and mutation rules

Every semantic value comes from `plan.json` or a versioned generated
configuration bound to it. Generated programs revalidate plan and candidate
identity before use.

The policy snapshot has five current registered policy rows and one canonical
digest chain for the complete ordered snapshot:

- `policy/model-policy-snapshot.v1.json` contains the canonical snapshot bytes;
- `plan.json` binds them as `model_policy_snapshot_sha256`;
- `bundle-manifest.json` repeats the digest as `policy_snapshot_sha256`; and
- the manifest file entry binds the same path, size, and digest.

Package-free validation proves this frozen-snapshot integrity and reproduces the
saved decision. It cannot prove current host policy. Installed-host validation
adds the fourth, current-registry digest comparison; a coherent v5 plan that is
no longer current requires replanning and a newly compiled bundle.

The current path identities include
`mlx-lm.qlora.single.attention-qkvo.v1` with profile `attention-qkvo.v1` for
Qwen3 MoE, `mlx-lm.qlora.single.dense-causal-lm.v1` with profile
`dense-causal-lm.v1` for dense Qwen2, `mlx-lm.qlora.single.gemma4-dense.v1` for
dense Gemma 4, `mlx-lm.qlora.single.gemma4-unified.v1` for unified Gemma 4
(Exit B / compiler-contract unsupported), and
`mlx-lm.qlora.single.gemma4-moe.v1` with profile `attention-qkvo.v1` for
Gemma 4 MoE. Serialization is not runtime support. The dense Qwen2 row binds a
uniform four-bit, group-size-64 layout with no module overrides and targets
q/k/v/o plus gate/up/down projections. The
[2026-08-05 Qwen2 MLX-LM exact-source refresh](../operations/evidence/2026-08-05-qwen2-mlx-lm-exact-source-refresh/README.md)
records two fresh, clean current v5/v3 `measured-run-pass` repetitions for the
exact pinned artifact, source commit
`719255153e3fc7e38e83b5ff826d587e5e58bf80`, source tree, Apple M5 Pro host,
Python/MLX runtime, dataset, policy snapshot, and bundle fingerprint
`ca2548cf8469fb9867f1558428803b1c9f7c19f48cba754fdb602643f23d1919`.
Relative to the unchanged [original acceptance
baseline](../operations/evidence/2026-08-05-qwen2-mlx-lm-acceptance/README.md),
the manifested generated `README.md` and `runbook.md` are the only changed
bundle paths; runtime programs and requirements remain byte-identical.
Generated-code changes that affect any of those bindings
require renewed evidence. The policy is still a configuration footprint rather
than an artifact allowlist; another matching artifact remains gated, and the
record does not qualify CUDA or establish safety, model quality, performance,
production throughput, production readiness, or release readiness.

This additive registry change does not rename the surrounding contracts. Keep
`aptus.model-policy-snapshot.v1`, `aptus.training-plan.v6`, and
`aptus.bundle.v3` unless the serialized shapes themselves change.

When adding a compiler-managed file:

1. write it before manifest creation;
2. include it in the expected required-file set when mandatory;
3. add path, digest, and size validation;
4. decide whether the deterministic ZIP includes it;
5. add missing, modified, symlink, and unexpected-file tests.

When adding a runtime output, keep it outside the compiler-managed file list and
define a runtime schema, producer, verifier, binding, manifest coverage, and
cleanup rule.

## Keep parent and child authority separate

The training child can write pending metrics and export evidence. It cannot
promote its own job to success. The managed or portable parent must verify:

- aggregate process exit;
- job, run, plan, candidate, placement, and rank bindings;
- finite losses and positive completed steps;
- measured CUDA peaks;
- trainable census and exact optimizer membership;
- dataset split and cross-rank agreement;
- expected export form and recursive file manifest.

A generated-code change that adds child evidence must add independent
parent-side rejection tests for missing, stale, malformed, and misbound values.
Managed admission, worker launch, and the completion verification and promotion
transaction must also retain their current-host model-policy checks. A portable
parent cannot substitute its embedded snapshot for that host authority.

## Dependency changes

`requirements.txt` contains exact direct pins selected by method. It is not a
complete transitive lock. When changing a pin:

- update `STACK_VERSIONS` and method dependency selection;
- prove clean installation;
- record the resolved runtime distribution closure;
- rerun model load, preflight, pilot, checkpoint, export, and cancellation
  paths affected by the dependency;
- update evidence and release records.

A library release note is not proof that the generated path still behaves the
same way.

## Minimum test coverage

Run at least:

```bash
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m unittest tests.aptus.test_generation -v
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m unittest tests.aptus.test_validation -v
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m unittest tests.aptus.test_execution -v
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m unittest tests.aptus.test_policy_snapshot -v
```

Then run the complete repository gate. Compiler changes also require a clean
wheel build and installed-wheel smoke. `tests.aptus.test_packaging` verifies
wheel package-data declarations and frozen-sidecar collection. Generation tests
compare emitted program bytes and manifest hashes with the packaged resources.
Runtime-semantic changes require the applicable real target-host evidence before
a release support claim. CUDA changes require the affected CUDA pilots. MLX-LM
changes require the affected dependency, model-data, measured-preflight, pilot,
full-run, and fresh-reload evidence.

## Review the generated diff

Check for:

- unexpected package or network dependencies;
- unbound defaults or changed precision behavior;
- data copied beyond documented paths;
- changed masking, splitting, checkpoint, or export semantics;
- shell interpolation or unsafe path handling;
- missing cancellation and lease participation;
- weakened finite-value, census, identity, or parent-verification checks;
- changed policy semantics without host/portable parity, canonical snapshot,
  digest-binding, stale-host, and package-free regression coverage;
- changed bundle files without manifest or reference updates.

Include the generated-tree diff or a concise manifest diff in the pull request.

## Related documentation

- [Artifact compiler](../architecture/artifact-compiler.md)
- [Bundle manifest](../reference/bundle-manifest.md)
- [Changing contracts](changing-contracts.md)
- [Execution orchestrator](../architecture/execution-orchestrator.md)
- [Release gates](../operations/release-gates.md)
