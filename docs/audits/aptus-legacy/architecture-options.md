# Aptus Architecture Options After the Legacy Audit

> **Documentation status:** Archived evidence
>
> **Applies to:** Dated forensic review of the removed legacy `HyperTune/` tree
>
> **Last reviewed:** 2026-08-06
>
> **Next scheduled review:** 2027-08-06, or when provenance or reproduction paths change
>
> **Historical warning:** The body below is preserved point-in-time evidence,
> not current architecture guidance. Start with the [audit index](README.md) or
> [current capabilities](../../product/current-capabilities.md).

This document does not authorize implementation. It frames the next design
decision using evidence recovered from the legacy `HyperTune/` folder.

## Non-negotiable boundaries

All viable options should preserve these boundaries:

1. **Typed inputs:** `HardwareSpec`, `DatasetProfile`, `ModelSpec`, and
   `TrainingTarget`, with explicit units and provenance.
2. **Pure decision core:** feasibility, resource strategy, method ranking, and
   hyperparameter selection must be deterministic and testable without
   importing Torch or downloading a model.
3. **Calibrated estimates:** predicted VRAM and time must identify their model,
   hardware, framework, version, confidence interval, and calibration source.
4. **Artifact materialization:** generated Python/configuration is separate from
   planning logic.
5. **Validation before release:** no generated artifact is returned as
   “ready-to-run” until it passes parse, dependency, framework-contract, and
   bounded smoke checks.
6. **Thin interfaces:** CLI, API, and MCP are adapters over one core contract,
   not competing implementations.

The legacy system repeatedly crossed these boundaries. Its TypeScript and
Python optimizers duplicated formulas, generated code without validating it,
and mixed provider, billing, and deployment concerns into an unproven core.

## Option A: Python-first planning engine

Build Aptus as a typed Python package with a local CLI first. Add FastAPI and
MCP adapters only after the core contract is stable.

Core flow:

`inputs → model/dataset/hardware profiles → feasibility planner → ranked strategy
→ hyperparameter plan → artifact generator → validator`

Why this fits the evidence:

- Fine-tuning libraries and the strongest salvageable runtime concepts are
  Python-native.
- Legacy `resource_scanner.py`, `dataset_analyzer.py`, model/method tables, and
  LoRA trainer/evaluator references can be adapted without a language boundary.
- A pure planner can avoid importing Torch until validation or execution.
- Pydantic models can make units, assumptions, warnings, and confidence
  machine-readable.

Costs and risks:

- MCP and web clients need adapters.
- GPU/framework compatibility still requires a versioned capability registry.
- Heuristic formulas must be re-derived and calibrated; Python does not make
  them correct automatically.

Best when: the first Aptus milestone must prove the central promise with a
trustworthy local plan and validated training bundle.

## Option B: TypeScript control plane with Python worker

Use TypeScript for MCP/API/workflow orchestration and a separate Python process
for model inspection, planning, and generated-script validation.

Why it may fit:

- Strong MCP and SaaS typing on the control plane.
- Clear process boundary for untrusted model/framework operations.
- Natural expansion path for hosted jobs, billing, and provider integrations.

Costs and risks:

- Two schemas, two dependency graphs, serialization, process management, and
  cross-language debugging from day one.
- The legacy project demonstrates exactly how quickly this becomes competing
  sources of truth.
- No legacy TypeScript entrypoint or method registry is reusable as-is.

Best when: a hosted MCP/API product is the immediate goal and the team can
support a deliberate polyglot architecture.

## Option C: Framework-config compiler

Narrow Aptus to profiling inputs and emitting validated configurations for one
established training framework, such as Transformers/TRL, Axolotl, or
LLaMA-Factory. Defer broad method ranking and job execution.

Why it may fit:

- Fastest route to a useful, testable artifact.
- Delegates distributed training and framework details to an existing project.
- Validation can focus on a bounded schema/version matrix.

Costs and risks:

- Smaller differentiator: Aptus becomes a configuration compiler unless the
  planner and calibration dataset later become substantive.
- Framework version drift remains significant.
- Supporting multiple targets too early recreates the legacy template sprawl.

Best when: near-term utility matters more than implementing the full
orchestration vision immediately.

## Evidence-based recommendation for the next design session

Start the next design session from Option A, but constrain its first vertical
slice like Option C: one Python core, one CLI, one model-family path, LoRA and
QLoRA, one generated Transformers/PEFT bundle, and validation through config
construction plus a tiny offline smoke test.

This is a recommendation, not an implementation decision. The user should
approve the runtime, first model/method matrix, validation threshold, and
calibration strategy in a separate Aptus design.
