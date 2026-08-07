# Aptus Legacy Recovery Audit: Executive Summary

> **Documentation status:** Archived evidence
>
> **Applies to:** Dated forensic review of the removed legacy `HyperTune/` tree
>
> **Last reviewed:** 2026-08-06
>
> **Next scheduled review:** 2027-08-06, or when provenance or reproduction paths change
>
> **Historical warning:** The body below is preserved point-in-time evidence,
> not current product guidance. Start with the [audit index](README.md) or
> [current capabilities](../../product/current-capabilities.md).

Date: 2026-07-21

## Bottom line

The legacy `HyperTune/` folder is not a functioning fine-tuning orchestrator.
It is a substantial product concept and research prototype containing useful
domain knowledge, several salvageable implementation seams, and a large amount
of duplicated or unfinished scaffolding.

The project reached:

- a clear user workflow;
- multiple competing architecture sketches;
- model-, method-, task-, dataset-, and hardware-aware heuristics;
- partial Python training and analysis utilities;
- draft CLI, API, MCP, provider, deployment, auth, and billing surfaces.

It did not reach:

- a reproducible build in either language;
- one coherent runtime or source of truth;
- a valid end-to-end optimizer;
- a working script generator;
- calibrated VRAM/time estimates;
- a real precision and quantization decision engine;
- credible automated tests;
- a generated artifact proven to run.

The right recovery strategy is therefore not “finish the old app.” Aptus should
preserve and re-verify the knowledge, then implement the core promise behind
clean typed boundaries.

## What the audit proved

- 228 legacy artifacts totaling 1,879,017 bytes were hashed and inventoried.
- 38 exact duplicate clusters contain 98 files.
- 23 files are byte-empty.
- The tree contains 143 script files: 71 Python and 72 JavaScript/TypeScript.
- Static analysis found three Python parse failures and 40 missing relative
  imports.
- Nine bounded checks produced two passes, six failures, and one blocked test-collection gate. They ran from Cursor's sandbox, while the audit runner itself records that it does not enforce OS-level isolation.
- Strict classification produced:
  - 35 ADAPT candidates;
  - 126 ARCHIVE artifacts;
  - 67 DISCARD candidates;
  - zero direct KEEP artifacts.
- A high-confidence token/private-key scan found zero matches. This is not proof
  that every file is safe to publish.
- The baseline and current legacy source hashes remain identical. No legacy
  file was edited or deleted.

Zero KEEP does not mean zero value. Under this audit’s rubric, KEEP requires a
unique artifact to build, resolve, run, and pass relevant checks as-is. Many
valuable files are ADAPT because their ideas survive while their implementation
does not.

## What was right

### The product problem

The central idea is still sound: take a model, dataset, hardware profile, and
training objective; determine a feasible strategy; and emit a validated,
ready-to-run training bundle. The root Aptus README states this more precisely
than most of the legacy material.

### The conceptual pipeline

`SystemArchitecture.md` identifies useful responsibilities: input/API,
optimization core, explanation, artifact generation, and formatting.
`README.md`, `src/python/hyperparameter_mcp.py`, and the FastAPI drafts converge
on the workflow:

`analyze model → profile dataset → scan hardware → plan → generate`

That flow should survive. The legacy implementations should not.

### The useful domain decomposition

The strongest legacy work is the recognition that recommendations depend on:

- model family, size, hidden width, layer count, and target modules;
- dataset size, token count, length distribution, and conversational format;
- device type, VRAM, BF16 support, and GPU count;
- task and quality/speed/memory priorities;
- fine-tuning method, quantization, batch size, accumulation, and sequence
  length.

This decomposition is the basis of a real Aptus planner.

## Hidden gems worth extracting

The detailed ranking is in `hidden-gems.md`. The highest-value candidates are:

1. `src/python/resource_scanner.py` — cross-platform hardware discovery. Its
   as-is smoke test stopped at a missing dependency; a disposable dependency
   shim completed the scanner contract. Rewrite and test it rather than copying.
2. `src/python/dataset_analyzer.py` — dataset format and length profiling,
   sampling, token estimation, and validation-split guidance. The contract and
   naming need correction.
3. `src/hypertuner/task-configs.ts`, `src/method-constraints.ts`, and
   `src/formulas/target_modules.ts` — useful seed tables for method/task/model
   compatibility. They are research priors, not verified “optimal” settings.
4. `src/model-database.ts` and `src/model-database-update.ts` — a useful
   registry concept and model-family metadata, subject to source validation.
5. `src/python/core_optimizer.py` — practitioner explanations, search ranges,
   target-module mappings, BF16 checks, and rough memory components. Extract the
   knowledge; discard Optuna over synthetic objectives.
6. `src/hypertuner/training/lora_trainer.py` and
   `src/hypertuner/evaluation/lora_evaluator.py` — the cleanest reference shapes
   for a future generated LoRA bundle, but not yet sandbox-proven.
7. `docs/reft_methods_guide.md` and the vendored PyReFT examples — useful
   research context. The current upstream project is Apache-2.0, but the
   vendored snapshot's exact revision, modifications, and license coverage are
   unresolved; it must remain outside Aptus-owned source.
8. The explanation-engine idea — every recommendation should report why it was
   selected, its assumptions, predicted resource use, confidence, and
   alternatives.

## What is broken, misleading, or trash

### Broken core mechanics

- `src/python/core_optimizer.py` does not parse.
- Its Optuna objectives optimize deterministic, monotonic hand-written scores
  rather than training or evaluation results. “Expected performance” is
  synthetic.
- `src/python/script_generator.py` does not parse.
- `src/python/script_generator_v2.py` omits Jinja2 from its requirements and
  calls template methods that do not exist.
- The TypeScript tree fails parsing/type-checking and imports a missing method
  registry and missing resource estimator.
- `server.js` parses, but npm cannot produce a lockfile from its manifest.

### Claims that must not ship

- Hard-coded fake training/evaluation metrics in `src/python/tune_service.py`.
- Speculative parameter counts for closed GPT, Claude, and Gemini models.
- Placeholder “premium” methods and model files presented as product features.
- Universal claims derived from task-specific ReFT or QLoRA experiments.
- Uncalibrated heuristics described as “optimal” or as expected accuracy.
- Placeholder legal documents and inconsistent MIT/proprietary assertions.

### Proven cleanup candidates

- Exact Finder/iCloud `* 2*` copies.
- Duplicate test and `src/python 2/` trees.
- Repeated deployment guides and configurations.
- Empty feature/API/schema files.
- `.DS_Store` artifacts.
- Generic unfilled legal templates.

The 67 DISCARD labels are recommendations only. Nothing was deleted.

## What remains to build for Aptus

The differentiator described in the Aptus vision is mostly absent from the
legacy code. Aptus still needs:

1. **Typed facts:** explicit hardware, model, dataset, target, strategy, plan,
   estimate, warning, and artifact schemas with units and provenance.
2. **A real resource planner:** component-level accounting for weights,
   quantization metadata, gradients, optimizer states, activations, temporary
   kernels, fragmentation, checkpointing, offload, sharding, and safety margin.
3. **Precision selection:** hardware/framework-aware FP32, FP16, BF16, and later
   FP8 decisions rather than a fixed model-family default.
4. **Strategy ranking:** deterministic feasibility first, transparent scoring
   second, and optional empirical search only when it measures real runs.
5. **A capability registry:** verified model families, target modules,
   framework versions, kernels, devices, and known incompatibilities.
6. **Artifact generation:** versioned templates for one bounded framework and
   method matrix.
7. **Artifact validation:** parse, dependency, schema, framework-construction,
   and tiny offline training gates before calling output ready-to-run.
8. **Calibration:** benchmark fixtures and measured VRAM/throughput records that
   turn estimates into testable predictions.

## Recommended next step

Use a new Aptus Python-first core and treat the legacy folder as an evidence
archive. Begin with a narrow vertical slice:

- local CLI;
- one open model family;
- LoRA and QLoRA;
- deterministic hardware/dataset/model profiling;
- a transparent strategy decision;
- one Transformers/PEFT training bundle;
- parse and config-construction validation;
- one tiny offline smoke train.

Do not add hosted providers, billing, premium gating, deployment matrices, or a
web control plane until that slice predicts resources credibly and emits a
working artifact.

The three architecture paths and trade-offs are in
`architecture-options.md`. Any extraction, deletion, or Aptus implementation
should begin only after a separate design is approved.
