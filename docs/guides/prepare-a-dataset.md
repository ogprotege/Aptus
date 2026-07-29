# Prepare a Dataset

> **Status:** Active | **Audience:** Fine-tuning practitioners and data reviewers | **Authority:** Operational | **Applies to:** Aptus 0.2 | **Owner:** Data contracts | **Last reviewed:** 2026-07-28 | **Review by:** 2026-10-22

Aptus 0.2 accepts supervised fine-tuning data from local JSONL, JSON, CSV, or
text files. Preparation has two separate goals: make every row structurally
valid, and make the corpus suitable for the task you intend to evaluate.
Aptus enforces the first goal. Human review and an explicit evaluation design
remain necessary for the second.

## Accepted file containers

| Suffix | Container contract |
|---|---|
| `.jsonl` | One JSON object per nonblank line |
| `.json` | One object, a list of objects, or an object whose `train` field is a list |
| `.csv` | A header row followed by records parsed by Python's `csv.DictReader` |
| `.txt` | One line becomes one `{"text": "..."}` row. Not compilable for `mlx-lm`; see [Whole-text supervision](#whole-text-supervision) |

Every usable row must match one of the schemas below. A file can contain more
than one accepted schema. Aptus records such a profile as `mixed`.

## Accepted row schemas

### Whole-text supervision

```json
{"text": "The complete sequence is supervised."}
```

`content` is accepted as an alternative whole-text field:

```json
{"content": "This complete sequence is also supervised."}
```

Whole-text rows train on every retained token.

**Whole-text rows do not compile for `mlx-lm`.** Pinned MLX-LM 0.31.3 cannot
combine full-text supervision with the bundle's required prompt masking, so the
compiler refuses both `text` and `content`-only rows for that runtime. This is a
compile-time refusal: an Apple Silicon target cannot produce a bundle from a
whole-text corpus at all, including any dataset built from a `.txt` container.
For an Apple Silicon target, use prompt/completion, instruction/output, or
messages rows. Whole-text supervision remains supported on
`transformers-peft-cuda`.

### Prompt and completion

```json
{
  "prompt": "State the safety boundary.",
  "completion": "A plan is an estimate until target-host validation passes."
}
```

Both fields must be strings and the completion must be non-empty. The generated
trainer masks prompt tokens and applies loss to the completion.

### Instruction, optional input, and output

```json
{
  "instruction": "Summarize the finding.",
  "input": "The pilot exceeded its memory reserve.",
  "output": "The candidate is not authorized for full training."
}
```

`instruction` and `output` must be strings. `output` must be non-empty. `input`
is optional. Runtime transformation emits explicit Instruction, Input, and
Response sections before tokenization.

### Messages

```json
{
  "messages": [
    {"role": "user", "content": "What did validation prove?"},
    {"role": "assistant", "content": "It proved the recorded operational contract."}
  ]
}
```

Every message needs string `role` and `content` fields. The final message must
be a non-empty assistant message. Model-data validation applies the pinned
tokenizer's chat template and requires the prompt to be prefix-separable from
the completed conversation. Aptus refuses a template that cannot preserve
assistant-only masking without altering control tokens.

## Keep related material on one split side

Use `split_group` when multiple rows come from the same source, document,
conversation, person, problem seed, or paraphrase family.

Top-level form:

```json
{
  "prompt": "Question from document A, chunk 1",
  "completion": "Reviewed answer",
  "split_group": "document-a"
}
```

Metadata form:

```json
{
  "prompt": "Question from document A, chunk 2",
  "completion": "Reviewed answer",
  "metadata": {"split_group": "document-a"}
}
```

If both locations are present, their non-empty string values must match. The
full trainer assigns every row with the same declared value to one side of the
train and evaluation boundary. The value itself is not published in split
metrics.

Ungrouped data uses an exact row-count split. Grouped data reaches the requested
row count when the group sizes and available ungrouped rows make it attainable.
Otherwise it selects the closest feasible size and records the target, realized
size, fraction, and row error. Never break a real source relationship merely to
obtain an exact fraction.

## Review the profiling evidence

Run:

```bash
aptus profile \
  --dataset ./data/training.jsonl \
  --sample-limit 512 \
  --sequence-length 1024 \
  --output ./aptus-work/dataset-profile.json
```

Review:

- source SHA-256 and byte size;
- valid, empty, and normalized-duplicate counts;
- schema counts;
- median, 95th percentile, and maximum sequence estimates;
- truncation count at the requested sequence length;
- whether token counts are tokenizer-measured or use the recorded
  four-characters-per-token estimate;
- sampled row indices and sample size.

The sample limit affects length statistics only. Compilation reopens the source,
checks its digest, validates every supported row, and writes every usable row to
`data/training.jsonl`.

## Understand truncation and loss masking

For prompt-completion, instruction-output, and messages rows, Aptus preserves
supervised completion tokens first. It then retains the prompt suffix that fits
the sequence limit. A row with no supervised tokens is rejected.

For whole-text rows, all retained tokens are supervised. This is materially
different from completion-only training. Select the row schema deliberately.

Sequence packing is unsupported in Aptus 0.2. Setting `packing=true` makes the
candidate unsupported.

## Complete the human review

Before compilation, verify:

- permission, license, consent, and intended use for every source;
- removal or controlled handling of secrets and personal data;
- source fidelity and correct attribution;
- consistent prompt and answer policy;
- no empty, contradictory, malformed, or machine-generated target accepted
  without review;
- no evaluation answer or label leaked into the prompt;
- deduplication across exact copies, paraphrases, chunks, and related sources;
- stable IDs and `split_group` values for related rows;
- a final test set kept outside the training JSONL.

Compilation creates cleartext copies of the source, canonical rows, pilot rows,
and ZIP archive. Runtime work adds caches, logs, checkpoints, metrics, and final
artifacts. Protect all of them according to the corpus sensitivity.

## When to re-profile and recompile

Re-profile after any source-byte change. Recompile to a new path after a data,
plan, method, sequence-length, or compiler change. Aptus rejects a source whose
digest changed after profiling and a bundle whose compiler-managed files were
edited.

## Related documentation

- [Model, dataset, and hardware facts](model-dataset-hardware.md)
- [Reviewed corpus contract](../reference/reviewed-corpus-contract.md)
- [Design an evaluation](design-an-evaluation.md)
- [Bundle manifest](../reference/bundle-manifest.md)
- [Security policy](../../SECURITY.md)
