# Generated Code and Bundle Changes

> **Status:** Active | **Audience:** Compiler and runtime contributors | **Authority:** Operational | **Applies to:** Aptus 0.2 | **Owner:** Artifact compiler | **Last reviewed:** 2026-07-27 | **Review by:** 2026-10-27

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
| `_bundle_programs/mlx/preflight.py` | `preflight.py` | MLX cumulative runtime action orchestration |
| `_bundle_programs/mlx/validate.py` | `validate.py` | MLX validation ladder, attestations, and fail-closed evidence checks |

The compiler also copies current package sources into:

- `plan_contract.py` for identity and manifest checks;
- `runtime_lease.py` for portable host-global coordination.

These programs must run from the bundle environment without importing the Aptus
application package.

## Other compiler-owned outputs

The same module emits:

- portable `plan.json` and fact profiles;
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
9. Run the affected target-runtime gate. Use the CUDA pilot for CUDA resources
   and the MLX-LM evidence ladder for MLX resources.
10. Update bundle, validation, capability, and operations documentation.

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

## Preserve identity and mutation rules

Every semantic value comes from `plan.json` or a versioned generated
configuration bound to it. Generated programs revalidate plan and candidate
identity before use.

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
- changed bundle files without manifest or reference updates.

Include the generated-tree diff or a concise manifest diff in the pull request.

## Related documentation

- [Artifact compiler](../architecture/artifact-compiler.md)
- [Bundle manifest](../reference/bundle-manifest.md)
- [Changing contracts](changing-contracts.md)
- [Execution orchestrator](../architecture/execution-orchestrator.md)
- [Release gates](../operations/release-gates.md)
