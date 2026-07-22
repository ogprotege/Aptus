# Contributor Guide

> **Status:** Active | **Audience:** Contributors | **Authority:** Operational | **Applies to:** Aptus 0.2 | **Owner:** Maintainers | **Last reviewed:** 2026-07-22 | **Review by:** 2026-10-22

Aptus changes must keep product behavior, generated artifacts, evidence claims,
interfaces, tests, and documentation aligned. Begin with the smallest contract
you intend to change, then follow every consumer of that contract.

## Choose the right guide

| Change | Guide |
|---|---|
| Learn the repository | [Code map](../architecture/code-map.md) |
| Add or promote a fine-tuning method | [Adding a method](adding-a-method.md) |
| Change a schema, identity, state, or binding | [Changing contracts](changing-contracts.md) |
| Change bundle files or portable programs | [Generated code](generated-code.md) |
| Change the React application | [Workbench development](workbench.md) |
| Change runtime jobs or completion | [Execution orchestrator](../architecture/execution-orchestrator.md) |
| Change security-sensitive behavior | [Security boundaries](../architecture/security-boundaries.md) |

The root [CONTRIBUTING.md](../../CONTRIBUTING.md) remains the concise source for
setup, required checks, design rules, claim rules, and pull-request content.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[server,test]'
cd web
npm ci
```

Use Python 3.11 or newer. Node.js is required only for web development and for
rebuilding the workbench package assets.

## Repository-wide quality gate

From the repository root:

```bash
.venv/bin/ruff format --check src/aptus tests/aptus
.venv/bin/ruff check src tests tools
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m unittest discover -s tests -t . -v
PYTHONPATH=src .venv/bin/python -m compileall -q src tests tools
git diff --check
cd web
npm test
npm run typecheck
npm run build
```

Run the wheel build and installed-wheel workbench smoke test when source
packaging, package data, imports, CLI entrypoints, API serving, or web assets can
change. The CI workflow is the executable reference for that gate.

Repository tests do not replace target-host evidence. Run real CUDA pilots for
changes to method preparation, dependencies, precision, quantization,
distribution, memory accounting, checkpointing, dataset transformation,
training, or export.

## Contract-first workflow

1. Name the current authority and every affected consumer.
2. Decide whether the change is semantic and requires a version or identity
   change.
3. Add the failure tests first for any safety or unsupported boundary.
4. Change the typed contract and pure behavior.
5. Update generated runtime behavior when the portable bundle consumes it.
6. Update API, CLI, and web models together.
7. Compile a fresh bundle and inspect the emitted diff.
8. Update the canonical documentation and remove duplicated claims.
9. Run checks proportional to the risk.
10. Record evidence still missing from the target host.

## Evidence and claim rules

Keep these distinctions explicit:

- a paper defines or evaluates a method in its stated scope;
- provider metadata declares repository facts;
- the planner produces analytic decisions and estimates;
- preflight and pilot measure the exact selected path on one host state;
- parent verification attests a completed run and structural export;
- an explicit evaluation supports a task-quality claim.

Do not turn one level into another. In particular, avoid “optimal,” “guaranteed
fit,” “release-ready,” and “quality passed” unless the exact comparison and
evidence contract supports the words.

## Preserve fail-closed behavior

Aptus should return an explicit unsupported or invalid result when it lacks a
safe rule. Do not repair missing facts by silently changing the method,
sequence length, effective batch, precision, distribution, or hardware.

Keep these boundaries unless the change defines and proves a replacement:

- execution uses the Transformers and PEFT CUDA compiler or the separate
  single-device MLX-LM LoRA and QLoRA compiler; PyTorch MPS has no compiler;
- SFT is the only objective;
- full FP16, full FSDP, quantized FSDP, MLX full-parameter training, MLX DoRA,
  and packing are unsupported;
- full-training resume is unsupported;
- generated requirements are direct pins, not a transitive lock;
- the API is a trusted-user local interface;
- child exit cannot certify full-run completion;
- structural export verification is not quality evaluation.

## Pull-request evidence

Describe:

- the changed contract and its prior behavior;
- source modules and generated files affected;
- identity or schema-version decision;
- positive and negative tests run;
- compiled-bundle diff when generation changed;
- target hardware, model revision, data digest, and pilot result when runtime
  evidence was collected;
- current limitations and unpassed gates;
- documentation pages updated.

Never commit tokens, private datasets, caches, checkpoints, model weights, raw
job state, or personal machine artifacts.

## Related documentation

- [Root contributing guide](../../CONTRIBUTING.md)
- [System architecture](../architecture/system.md)
- [Current capabilities](../product/current-capabilities.md)
- [Release gates](../operations/release-gates.md)
- [Claim language](../product/claim-language.md)
