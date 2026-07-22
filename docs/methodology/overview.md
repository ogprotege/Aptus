# Methodology Overview

> **Status:** Active | **Authority:** Normative methodology index | **Applies to:** Aptus 0.2 | **Audience:** Practitioners, researchers, and contributors | **Last reviewed:** 2026-07-22 | **Review by:** 2027-01-22 or when a methodology version changes

Methodology version: `aptus-methodology-v2`.

Aptus separates factual resolution, hard feasibility, resource estimation,
preference ranking, artifact compilation, and measured validation. No later
stage may repair missing facts silently.

## Pipeline

1. Resolve explicit model, dataset, hardware, and target facts.
2. Profile dataset rows with a tokenizer when one is supplied, or with the
   recorded character-count estimate.
3. Resolve the selectable gated-executable methods from the typed registry.
4. Enumerate the fixed 12-candidate method-placement set.
5. Apply capability and policy constraints.
6. Calculate point and upper resource envelopes.
7. Annotate the viable Pareto frontier.
8. Rank every viable candidate under the recorded lexicographic objective.
9. Compile one immutable plan into an atomic bundle.
10. Validate the exact bundle through explicit levels.
11. Run a real-model pilot on the target hardware.
12. Require approval before full execution.
13. Record observed behavior for later calibration.

## Separation of concerns

Hardware determines which candidates may fit. It does not determine model
quality. Research and prior runs can suggest starting hyperparameters. Only
evaluation on the target task establishes quality.

The pure planner does not import Torch or download a model. Runtime profilers,
preflight workers, and job workers perform those operations behind explicit
boundaries.

## Methodology contracts

| Contract | Version | Purpose |
| --- | --- | --- |
| Facts | `aptus.facts.v2` | Explicit values and available provenance |
| Method descriptor | `aptus.method-descriptor.v1` | Runtime lifecycle, selectability, compiler, export, evidence, and blocker metadata |
| Candidates | `aptus-candidates-v2` | Finite strategy enumeration and constraint results |
| Precision | `aptus-precision-v2` | Compute and quantization selection policy |
| Memory | `aptus-memory-v2` | Per-device point and upper VRAM envelopes |
| Ranking | `aptus-ranking-v2` | Pareto annotation and lexicographic objective policy |
| Preflight | `aptus-preflight-v2` | Runtime-specific bounded update evidence before the real pilot; CUDA uses a synthetic model, while MLX uses the pinned model and compiled data |
| Bundle | `aptus.bundle.v2` | Atomic file manifest and execution contract |
| Plan | `aptus.training-plan.v2` | Selected candidate and full decision trace |
| Trainable census | `aptus.trainable-parameter-census.v1` | Method-scope tensor and parameter counts, finite state, and descriptor digest |
| Dataset split | `aptus.dataset-split.v1` | Full-run assignment strategy, counts, canonical digest, assignment digest, and realized error |
| MLX dataset split | `aptus.mlx-split.v1` | Disjoint compiled train and validation counts with repeat-within-split microbatch padding |

The plan records its schema and memory-formula versions. Related documentation
names the other rule sets. A changed execution-affecting equation or rule
requires a new version or an explicitly compatible patch.

## Abstention

Aptus marks candidates `unsupported` or raises a no-feasible-plan error when:

- a required input fails its contract;
- a model family has no target-module catalog entry;
- a runtime combination has no capability rule;
- every candidate violates a hard constraint.

V0.2 does not emit a separate `insufficient-evidence` candidate status.
Abstention remains the safe result when the current contracts cannot support a
recommendation.

## Primary references

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [Transformers](https://huggingface.co/docs/transformers/index)
- [PEFT](https://huggingface.co/docs/peft/index)
- [Accelerate](https://huggingface.co/docs/accelerate/index)
- [PyTorch FSDP](https://docs.pytorch.org/docs/stable/fsdp.html)
- [PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html)

## Related documentation

- [Method selection guide](../guides/choose-a-method.md)
- [Method registry](../reference/method-registry.md)
- [System architecture](../architecture/system.md)
- [Validation states](../reference/validation-states.md)
