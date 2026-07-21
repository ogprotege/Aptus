# Aptus Core Vertical Slice

Date: 2026-07-21

## Goal

Build the first trustworthy Aptus workflow:

1. accept an open model specification, local dataset, hardware constraints, and
   optimization target;
2. profile those inputs deterministically;
3. compare LoRA and QLoRA using explicit feasibility rules and transparent,
   component-level memory estimates;
4. produce a typed training plan with assumptions, warnings, alternatives, and
   evidence labels;
5. generate a versioned Transformers/PEFT training bundle;
6. reject the bundle unless it passes static and offline smoke validation.

This slice proves the product contract. It does not claim universal optimality
or calibrated prediction accuracy.

## Product principles

- Aptus is the only product name.
- Unknown facts fail closed; they do not silently default to Llama, LoRA, or a
  guessed model size.
- “Recommended” always means “under these listed assumptions.”
- Hard feasibility constraints are separate from preference scoring.
- Every estimate reports its components, units, safety margin, method, and
  confidence.
- Generated code consumes `plan.json`; user paths and identifiers are never
  interpolated into executable source.
- The decision core does not import Torch, Transformers, PEFT, or access the
  network.
- No output is called ready-to-run without a validation report.

## Scope

Supported:

- local `.json`, `.jsonl`, `.csv`, and `.txt` datasets;
- plain `text` examples for generated training;
- manually supplied open-model structural facts;
- Llama-, Mistral-, Gemma-, and Qwen-style target-module aliases;
- CUDA hardware profiles;
- LoRA and QLoRA;
- objectives `quality`, `memory`, and `speed`;
- one Transformers + PEFT bundle target;
- static validation and an optional fully offline tiny-model smoke step.

Deferred:

- remote model/dataset discovery;
- DPO, DoRA, ReFT, full fine-tuning, distributed sharding, CPU/MPS training;
- real job execution, RunPod/cloud providers, persistence, billing, API, MCP,
  and UI;
- learned or Bayesian optimization;
- production calibration claims.

## Package boundaries

```text
src/aptus/
  domain.py            Typed input, plan, estimate, and validation contracts
  profiling.py         Local dataset profiling and explicit model/hardware specs
  catalog.py           Versioned method and target-module priors
  planning.py          Pure feasibility, memory accounting, and ranking
  generation.py        Plan-to-bundle compiler
  validation.py        Bundle parse, schema, contract, and smoke checks
  cli.py               Thin command adapter
```

The modules communicate only through the contracts in `domain.py`.

## Input contracts

`ModelSpec`

- immutable model ID/revision;
- family;
- parameter count;
- hidden size;
- layer count;
- context length;
- explicit license/training-permission note.

`DatasetProfile`

- source hash and format;
- example count;
- estimated token count;
- p50/p95/max sequence lengths;
- detected schema;
- sampled/full measurement status;
- warnings.

`HardwareSpec`

- backend;
- GPU count;
- per-device total VRAM;
- BF16 capability;
- host RAM;
- explicit user safety reserve.

`TrainingTarget`

- objective;
- requested sequence length;
- target effective batch size;
- maximum epochs;
- optional method preference.

## Planner

The planner evaluates LoRA and QLoRA independently.

Memory components:

- base weights;
- quantization metadata;
- trainable adapter weights;
- adapter gradients;
- optimizer states;
- estimated activations;
- temporary/framework overhead;
- user reserve;
- safety margin.

The first activation model is deliberately labeled `heuristic-v1`. Its
assumptions are emitted in the plan and tested for monotonicity. It must never
be presented as calibrated peak VRAM.

Feasibility:

- compares peak estimate with per-device usable VRAM, never aggregate VRAM;
- QLoRA requires CUDA and 4-bit backend support;
- BF16 is selected only when explicitly supported;
- micro-batch is searched from largest to smallest;
- gradient accumulation derives the requested effective batch;
- unsupported combinations return rejection reasons.

Ranking:

- quality prefers feasible LoRA, then QLoRA;
- memory prefers the lower peak estimate;
- speed prefers the lower quantization/accumulation overhead;
- user method preference cannot override infeasibility.

## Generated bundle

```text
output/
  plan.json
  plan_contract.py
  train.py
  requirements.txt
  validate.py
  README.md
  validation-report.json
```

`train.py` reads `plan.json` and supports:

- `--validate-only`: validates plan and environment contracts before importing
  the training stack;
- `--smoke`: constructs a tiny causal model from local configuration, applies
  PEFT, runs one synthetic forward/backward/optimizer step, and performs no
  network access;
- normal training against the profiled local text dataset.

The bundle pins the dependency versions used by the smoke environment.

## Validation gates

1. Plan JSON schema and cross-field invariants.
2. Python AST for all generated Python.
3. Requirements and expected files.
4. No unresolved template markers.
5. No user value embedded in executable source.
6. Target modules present in the supported family catalog.
7. `train.py --validate-only`.
8. Optional `train.py --smoke` in an isolated environment with network disabled.

Validation states are `invalid`, `static-pass`, `environment-pass`, and
`smoke-pass`.

## Legacy extraction and deletion gate

`HyperTune/` is temporary source material, not a permanent archive inside
Aptus. Before deletion:

1. Keep the forensic inventory, hashes, provenance, hidden-gems report, and
   classification ledger.
2. Produce `docs/audits/aptus-legacy/extraction-ledger.md` mapping every ADAPT
   candidate to:
   - implemented in Aptus;
   - captured as a future requirement/research lead;
   - deliberately rejected, with reason.
3. Pass all Aptus unit/integration/generation tests.
4. Regenerate the audit evidence and confirm the legacy source manifest remains
   `44156dd49da5f283aa2761baca7eb614cb0b07475dc65289da4409f585367084`.
5. Delete `HyperTune/`.
6. Remove `/HyperTune/` from `.gitignore`.
7. Verify Git status contains only Aptus-owned source, tests, and documentation.

The user has confirmed a separate local backup exists and authorized this
eventual deletion.

## Acceptance criteria

- One CLI command creates a plan and validated bundle from local fixtures.
- Repeated identical inputs produce byte-identical plan content except explicit
  generation timestamps, if any.
- QLoRA is selected for a constrained CUDA profile where LoRA is infeasible.
- LoRA is selected for a quality target when both methods fit.
- No plan is produced when neither method fits.
- Every rejection and recommendation includes an explanation.
- All memory values use bytes internally and GiB only at display boundaries.
- Generated Python parses and `--validate-only` succeeds.
- Offline smoke reaches one optimizer step when the optional ML environment is
  available.
- The legacy folder is removed only after the extraction ledger and all gates
  pass.
