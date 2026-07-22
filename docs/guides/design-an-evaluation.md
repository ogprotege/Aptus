# Design an Evaluation

> **Status:** Active | **Audience:** Fine-tuning practitioners and researchers | **Authority:** Explanatory | **Applies to:** Aptus 0.2 | **Owner:** Evaluation research | **Last reviewed:** 2026-07-22 | **Review by:** 2026-10-22

Aptus 0.2 does not implement a first-class quality-evaluation contract. It can
verify training work, finite losses, checkpoint continuation, bindings, and the
structural export file tree. None of those proves that the adapted model is
better for the target task.

Use this guide to define an external evaluation before training. Keep its data
and decisions separate from the training bundle until Aptus has a versioned
evaluation feature.

## Write the claim first

A useful evaluation starts with one bounded statement:

> On dataset D at immutable revision R, artifact A must improve metric M over
> baseline B by threshold T in direction X, under protocol P.

If the claim cannot name those terms, the evaluation cannot support it.

## Required evaluation record

| Field | Required decision |
|---|---|
| Task | Exact behavior being tested |
| Test dataset | Immutable identity, digest, rights, and inclusion criteria |
| Unit of analysis | Row, conversation, document, person, problem, or another independent unit |
| Baseline | Pinned untouched base model and any other comparison artifacts |
| Metric | Definition, implementation version, direction, and aggregation |
| Threshold | Minimum acceptable absolute result or improvement |
| Inference protocol | Prompt template, decoding settings, seed policy, context, and tools |
| Repetitions | Seeds, reruns, confidence interval, or other uncertainty treatment |
| Human rubric | Dimensions, scale anchors, reviewer qualifications, and adjudication |
| Failure policy | Missing output, refusal, timeout, parse failure, and non-finite handling |
| Contamination policy | How overlap, paraphrase leakage, and prior exposure are checked |
| Decision rule | Exact pass, fail, abstain, and tie behavior |

Bind the record to the model revision, adapter or model artifact digest, run ID,
dataset digest, code revision, environment, and evaluation timestamp.

## Separate three data roles

1. Training data changes model parameters.
2. Validation data informs model selection or stopping.
3. Test data supports the final comparison and must not influence training or
   selection.

The evaluation split created inside an Aptus full run is operational evidence
for loss reporting. It is not automatically a final test set. Keep the final
test set outside the training JSONL and freeze it before comparing methods.

Group all material from one source or seed before splitting. The same document,
conversation, person, problem template, or paraphrase family must not appear on
both sides. Record group-level overlap checks, not only exact string duplicate
checks.

## Compare methods fairly

When comparing full, LoRA, int8-LoRA, QLoRA, or a future method, hold constant
everything that is not the subject of the comparison:

- immutable base model and tokenizer revision;
- train, validation, and test assignments;
- prompt and loss-masking policy;
- sequence length and effective batch;
- training-token or step budget;
- checkpoint selection rule;
- evaluation prompts and decoding;
- metric implementation and threshold;
- random seeds where the method permits it.

Report resource results separately from quality results. A lower CUDA peak or a
faster step time is an efficiency result. It is not a quality improvement.

## Use metrics that match the task

Prefer deterministic task metrics when the desired output has a checkable
answer, such as exact match, structured-field accuracy, retrieval recall, or
program execution. Define normalization and partial-credit rules before seeing
the result.

For open-ended generation, use a rubric with narrow dimensions. Examples
include source fidelity, unsupported attribution, completeness, relevance,
format compliance, and harmful-error rate. Give every score point a behavioral
anchor. Record reviewer identity or role, blind the artifact label when
possible, and adjudicate material disagreement.

Do not compress distinct harms into one average. A model can improve general
helpfulness while worsening citation fabrication or unsafe compliance.

## Evaluate the untouched baseline

Run the exact protocol on the pinned base model before inspecting the adapted
result. If a production adapter or model already exists, include it as another
baseline. Store raw outputs so aggregate metrics remain auditable.

A training loss decrease and an evaluation loss decrease can guide diagnosis,
but neither replaces a task-specific comparison against the untouched base.

## Report uncertainty and exclusions

State the number of independent units, excluded rows, failed generations,
reviewer agreement, seeds, and confidence treatment. Preserve per-example
results with stable IDs when privacy permits. Report both aggregate and
high-severity slices.

Abstain when the test set is too small, contaminated, changed after the plan,
or evaluated with a protocol chosen after seeing results. An honest abstention
is stronger than a post-hoc success threshold.

## Minimum result packet

Preserve:

- evaluation specification and code revision;
- model, tokenizer, adapter, and run identities;
- test dataset and grouping digests;
- environment and hardware record;
- raw model outputs;
- per-example scores and reviewer records;
- aggregate metrics and uncertainty;
- pass, fail, or abstain decision with reasons;
- known limitations and excluded claims.

Do not write this packet into a compiler-managed bundle path. Store it as a
separate immutable evaluation artifact until Aptus defines an evaluation
manifest and verifier.

## Related documentation

- [Inspect results](inspect-results.md)
- [Ranking and uncertainty](../methodology/ranking-uncertainty.md)
- [Reviewed corpus contract](../reference/reviewed-corpus-contract.md)
- [Claim language](../product/claim-language.md)
- [Current capabilities](../product/current-capabilities.md)
