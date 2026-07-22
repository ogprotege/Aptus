# Adding a Fine-Tuning Method

> **Status:** Active | **Audience:** Planner and runtime contributors | **Authority:** Operational | **Applies to:** Aptus 0.2 | **Owner:** Method registry | **Last reviewed:** 2026-07-22 | **Review by:** 2026-10-22

A method name enters Aptus in stages. Research identity, runtime visibility,
planner selectability, compiler support, and release evidence are separate
states. Do not add a planner enum value or compiler branch merely because a
paper or dependency exposes a similarly named feature.

## Understand the two admission levels

### Research-visible, nonselectable

A nonselectable descriptor can document an accepted method identity and its
missing proof. It requires:

- a stable lowercase method ID and unambiguous mechanism;
- a primary source in the evidence registry;
- `experimental` or `research-only` lifecycle;
- `selectable=False`;
- no compiler ID or export contract;
- no supported backend or distribution;
- an explicit blocker;
- a concrete pilot requirement;
- aliases that do not collide with any ID or alias.

This descriptor can appear in the bootstrap method catalog and workbench
readiness board. It cannot appear in the planner preference control or candidate
matrix.

### Selectable, gated executable

A selectable method must also have a complete planning and execution contract.
It requires `gated-executable`, a unique compiler ID, an export kind, supported
backends and distributions, planner feasibility rules, resource estimates,
generated runtime code, validation, negative tests, and target-host pilot
evidence.

The registry validates that selectable IDs exactly equal the `Method` enum.
Adding one changes both contracts and expands the method by placement candidate
matrix. Treat that as a semantic planner change.

## Admission flow

```mermaid
flowchart LR
  P["Primary source"] --> E["Evidence record"]
  E --> D["Nonselectable descriptor"]
  D --> X["Typed config and compatibility"]
  X --> M["Method-specific estimates"]
  M --> C["Compiler and dependencies"]
  C --> T["Trainable-state contract"]
  T --> K["Checkpoint and restart contract"]
  K --> O["Export and reload verifier"]
  O --> N["Negative and integration tests"]
  N --> R["Real target-host pilots"]
  R --> S["Selectable descriptor"]
```

## 1. Establish the research identity

Add the primary claim to
[`src/aptus/evidence.py`](../../src/aptus/evidence.py). Record source kind,
scope, confidence, and revision. The evidence should define the mechanism, not
promise Aptus compatibility or quality.

Add or update the descriptor in
[`src/aptus/methods/registry.py`](../../src/aptus/methods/registry.py). Use
[`MethodDescriptor`](../../src/aptus/methods/contracts.py) fields literally.
Do not reuse `parameter_scope`, `parameterization`, or `base_storage` labels for
a materially different trainable object.

Add tests that every evidence ID resolves, lifecycle/selectability fields agree,
aliases are unique, and the blocker and pilot requirement are non-empty.

## 2. Define the objective and data contract

State the loss, required row schema, masking, sampling, partitions, and
evaluation needs. Aptus 0.2 compiles SFT only. Preference optimization,
continued pretraining, representation intervention, online RL, and bilevel
training need their own objective and data contracts.

Do not route a different objective through the SFT trainer because a library
uses a familiar class name.

## 3. Define typed configuration and compatibility

Every configuration field needs:

- type, bounds, unit, and default source;
- identity and schema-version treatment;
- supported model families and target types;
- precision and base-storage behavior;
- dependency and version requirement;
- supported placement and distributed ownership;
- explicit unsupported combinations.

Update [`domain.py`](../../src/aptus/domain.py) only when the method is ready to
join the selectable enum. Update API request models, CLI choices, web types, and
preference controls in the same change.

## 4. Add planner and memory behavior

Implement method-dispatched feasibility and resource accounting in
[`planning.py`](../../src/aptus/planning.py). Account separately for:

- base weights and quantization metadata;
- trainable tensors and gradients;
- optimizer and method-specific scheduler state;
- activations and checkpointing;
- communication, sharding, and ownership;
- workspaces, allocator reserve, and load transients;
- checkpoint retention and final export;
- host staging and disk.

Do not copy a LoRA coefficient for an adaptive, shared, sparse, or
representation-level method without proving the same state topology. Keep the
analytic point estimate and heuristic upper envelope separate. Increment the
formula version when execution-affecting semantics change.

## 5. Add a compiler contract

Define one unique compiler ID and one export kind. Update direct pins in
[`catalog.py`](../../src/aptus/catalog.py) and generated trainer configuration.
The bundle must expose the method identity, compiler ID, export kind, precision,
quantization, targets, batch, and distribution without hidden defaults.

Generated code must remain self-contained and portable. It must use the pinned
model revision and compiler-produced data. It must not download an unbound
revision, mutate the plan, or treat a library default as Aptus evidence.

## 6. Prove the trainable state

Define the exact eligible tensor set before optimizer construction. Require:

- unique names;
- positive tensor and parameter counts;
- finite initial values;
- a stable name-shape-dtype digest;
- exact optimizer parameter-ID membership;
- the same scope in model-data, measured preflight, pilot, and full training.

Method-specific topology needs method-specific validation. LoRA requires one
A/B pair per inspected target instance. BitFit needs a non-empty existing-bias
set. AdaLoRA needs changing-budget and importance state. ShareLoRA needs logical
versus unique parameter accounting. LoReFT needs intervention location and
token-position identity.

## 7. Define checkpoint and restart state

List every state required to continue the method exactly: model or adapter,
optimizer, scheduler, scaler, RNG, dataloader position, method-specific scores
or masks, distributed topology, environment, plan, and file manifest.

The current full-run resume boundary remains closed. A new method still must
prove the pilot semantics declared by its runtime. CUDA methods require
two-phase checkpoint continuation. An uninterrupted runtime must prove completed
updates, exact state scope, immutable output, and fresh-process artifact reload
without implying training resume. Do not expose general resume as a side effect
of adding method state files.

## 8. Define export and reload

Specify artifact files, provenance, tensor keys, index rules, base-model
binding, and recursive manifest coverage. Add a fresh-process reload test and a
bounded inference check before claiming semantic reload support. The current
CUDA verifier checks structural safetensors. MLX also performs a bounded
fresh-process adapter generation check. A new export form needs its own verifier
and evidence label.

## 9. Test every boundary

At minimum, cover:

- registry identity, lifecycle, alias, compiler, export, and evidence rules;
- all three placement rows and every unsupported combination;
- plan and candidate identity mutation;
- memory component arithmetic and upper bounds;
- deterministic bundle output and direct pins;
- generated source import and method preparation;
- empty, extra, non-finite, malformed, and mismatched trainable sets;
- optimizer membership;
- dependency, model-data, measured-preflight, and runtime-specific pilot work;
- checkpoint corruption and continuation failure where required, or
  uninterrupted-run and artifact-reload failure for MLX-style runtimes;
- export corruption, wrong provenance, and reload failure;
- single and distributed completion behavior;
- API bootstrap, planner preference, and workbench readiness state.

## 10. Collect release evidence

Run the exact compiled path in a clean environment on every claimed backend,
placement, precision, and quantization combination. Bind model revision, dataset
digest, package environment, hardware, plan, candidate, bundle, jobs, metrics,
runtime state artifacts, and exports.

Keep the descriptor nonselectable until code, tests, documentation, and the
required target-host evidence agree. A passing repository test on the
development Mac cannot promote a method.

## Related documentation

- [Fine-tuning method taxonomy](../methodology/method-taxonomy.md)
- [Method catalog](../methodology/method-catalog.json)
- [Candidate enumeration](../methodology/candidate-enumeration.md)
- [Generated code](generated-code.md)
- [Release gates](../operations/release-gates.md)
