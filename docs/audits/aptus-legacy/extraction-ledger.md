# Aptus Legacy Extraction Ledger

> **Documentation status:** Archived evidence
>
> **Applies to:** Dated forensic review of the removed legacy `HyperTune/` tree
>
> **Last reviewed:** 2026-08-06
>
> **Next scheduled review:** 2027-08-06, or when provenance or reproduction paths change
>
> **Historical warning:** The body below is preserved point-in-time evidence,
> not a current implementation ledger. Start with the [audit index](README.md)
> or [current capabilities](../../product/current-capabilities.md).

Date: 2026-07-21

This ledger closes the 35 ADAPT decisions in `classification.jsonl` before the
legacy `HyperTune/` folder is removed. “Implemented” means the useful contract
or idea was rebuilt in Aptus; it does not mean legacy source was copied.

## Rebuilt in the Aptus core

- `SystemArchitecture.md`
  - Preserved the separation among profiling, planning, explanation,
    generation, and validation in `docs/design/aptus-core-vertical-slice.md`
    and `src/aptus/`.
- `hypertune_cli.py`
  - Replaced by the typed, fail-closed `aptus plan` CLI in `src/aptus/cli.py`.
- `src/formulas/batch_size_v2.ts`
  - Replaced with per-device micro-batch search and explicit accumulation in
    `src/aptus/planning.py`. Legacy arithmetic was not copied.
- `src/formulas/target_modules.ts`
  - Reduced to a fail-closed, testable alias seed in `src/aptus/catalog.py`;
    unknown families now error instead of defaulting to Llama.
- `src/hypertuner/methodSelector.ts`
  - Replaced with feasibility-first candidate evaluation and transparent
    objective ranking in `src/aptus/planning.py`. Fixed accuracy scores were
    rejected.
- `src/hypertuner/training/lora_trainer.py`
  - Its real-training shape informed the generated `train.py`; precision,
    dataset, masking, configuration, and validation are now plan-driven.
- `src/optimizer.ts`
  - Preserved the filter → compare → recommend orchestration shape. Missing
    registries, premium gating, and unsafe output coupling were rejected.
- `src/output/command_line.ts`
  - Replaced shell-string generation with a structured bundle that reads
    `plan.json`.
- `src/output/config_file.ts`
  - Replaced ad hoc config emitters with deterministic typed JSON and a
    validation report.
- `src/python/config.py`
  - Replaced by immutable dataclass contracts in `src/aptus/domain.py`.
- `src/python/core_optimizer.py`
  - Preserved component-level memory vocabulary, method alternatives,
    explanations, and evidence labels. The parse failure, synthetic Optuna
    objective, “expected performance,” and unsafe memory math were rejected.
- `src/python/dataset_analyzer.py`
  - Rebuilt as deterministic local profiling in `src/aptus/profiling.py`, with
    source hashes, ordered percentiles, explicit estimation status, and
    fail-closed schemas.
- `src/python/script_generator_v2.py`
  - Replaced by `src/aptus/generation.py`; phantom templates, guessed CLI
    entrypoints, and unsafe YAML/shell output were rejected.
- `src/python/train.py`
  - Replaced by the generated plan-driven trainer. The new offline smoke path
    completed a real Transformers/PEFT optimizer step.

## Preserved as explicit future requirements or research leads

- `Integration_Architecture.md`
  - Provider/API/MCP/UI ambitions remain in
    `docs/audits/aptus-legacy/architecture-options.md`; none is claimed in the
    first slice.
- `api/FastAPI/main.py`
  - API request and workflow ideas are preserved for a later thin adapter.
    Unsafe uploads, local paths, and `trust_remote_code=True` are explicitly
    prohibited by the risk register.
- `docs/reft_methods_guide.md`
  - ReFT remains a provenance-scoped future method; exact upstream revision,
    dataset contract, and reproduction are required.
- `src/formulas/learning_rate_2.ts`
  - Constants are preserved only as unverified research hypotheses. Aptus uses
    an explicitly labeled conservative prior pending calibration.
- `src/formulas/rank_v2.ts`
  - Rank-scaling ideas remain research hypotheses; conflicting legacy formulas
    were not ported.
- `src/formulas/weight_decay_v2.ts`
  - Task/method multipliers remain unverified and were not ported.
- `src/hypertuner/evaluation/lora_evaluator.py`
  - The need for adapter reload, deterministic generation, perplexity, and task
    metrics is preserved. Exact-match sampled “accuracy” was rejected.
- `src/hypertuner/task-configs.ts`
  - Preserved as a future benchmark-observation catalog requirement, never as
    universal optimal defaults.
- `src/method-constraints.ts`
  - Preserved as the vocabulary for a future versioned capability registry.
    Fixed savings and wildcard compatibility were rejected.
- `src/model-database-update.ts`
  - Preserved requirements for source timestamps, field-level provenance,
    fail-closed updates, and atomic promotion.
- `src/model-database.ts`
  - Preserved the typed registry concept. Unattributed records and fuzzy silent
    selection were rejected.
- `src/python/export_model.py`
  - Adapter/model export and reload validation remain a future execution gate.
- `src/python/model_registry.py`
  - Preserved the need for immutable model revision identity and a verified
    catalog. The missing external database path was rejected.
- `src/python/register_dataset.py`
  - Dataset registration becomes a future storage/provider boundary; hard-coded
    global paths were rejected.
- `src/python/register_model.py`
  - Model registration becomes a future provider boundary with revision,
    checksum, license, and training-permission evidence.
- `src/python/resource_scanner.py`
  - Cross-platform discovery remains a next profiler. The first slice requires
    explicit hardware facts; aggregate VRAM, unit parsing, and batch-size bugs
    were not ported.
- `src/python/dora_decomposer.py`
  - DoRA mathematics is retained as a paper/reimplementation lead. The
    unregistered state and function-as-weight runtime were rejected.
- `src/python/flexora_optimizer.py`
  - Bilevel layer selection remains a research lead. The disconnected gradient
    implementation was rejected.
- `src/python/reft_adapter.py`
  - Preserved as a future upstream PyReFT integration requirement, not code.
- `src/python/reft_enhanced.py`
  - Layer/position configuration ideas remain research leads; fixed metrics and
    invalid dataset contract were rejected.
- `src/python/reft_setup.py`
  - The LoReFT equation remains a cited research lead; unregistered hook state
    and non-persistent runtime were rejected.

## Deletion authorization and evidence

- Every ADAPT path is listed above.
- ARCHIVE/DISCARD paths remain recoverable from the user's separate local copy.
- The forensic reports, hashes, duplicate clusters, provenance findings, and
  full 228-row classification remain in `docs/audits/aptus-legacy/`.
- The final pre-deletion legacy manifest is
  `44156dd49da5f283aa2761baca7eb614cb0b07475dc65289da4409f585367084`.
- The user explicitly authorized eventual removal after extraction.

## Completion

The Aptus test, smoke, manifest, extraction, and independent review gates
passed. The working `HyperTune/` folder and its `.gitignore` entry were removed
on 2026-07-21. This repository now contains only Aptus-owned product source,
tests, tooling, and the preserved forensic evidence.
