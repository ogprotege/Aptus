# Dataset Schemas and Transformation Contract

| Metadata | Value |
| --- | --- |
| Status | Active |
| Audience | Dataset authors, corpus reviewers, operators, and trainer maintainers |
| Authority | Normative v0.2 input, canonicalization, masking, and split reference |
| Last reviewed | 2026-07-22 |
| Next review | 2026-10-22, or sooner when profiling or generated data code changes |

Aptus accepts local supervised rows. It does not ingest a remote dataset name,
execute arbitrary formatting code, infer consent, or decide whether data is
appropriate to train on.

## Accepted file formats

| Suffix | File contract |
| --- | --- |
| `.jsonl` | One JSON object per nonblank line |
| `.json` | A list of objects, an object with a list-valued `train`, or one object |
| `.csv` | Header row plus records read by `csv.DictReader`; values are strings |
| `.txt` | One line per `{"text": "..."}` record |

Every row must resolve to an object. Unsupported suffixes and non-object rows
fail profiling. Blank JSONL lines are skipped.

## Schema detection order

For each object, Aptus tests supported shapes in this order:

1. `text`;
2. `prompt` and `completion`;
3. `instruction` and `output`;
4. `messages`; and
5. `content` as a text alias.

This precedence matters when a row contains fields from more than one schema.
The first recognized shape controls profiling and runtime transformation.

## Text schema

```json
{
  "text": "The complete supervised sequence."
}
```

Requirements:

- `text` is a string;
- after whitespace trimming, it is non-empty.

`content` is accepted as an alternative field with the same rules:

```json
{
  "content": "The complete supervised sequence."
}
```

Runtime behavior:

- the tokenizer encodes the full value with special tokens;
- all retained tokens are supervised;
- truncation keeps the tokenizer's first `sequence_length` tokens.

The canonical schema name for either field is `text`.

## Prompt-completion schema

```json
{
  "prompt": "Question: What is Aptus?\nAnswer:",
  "completion": " An evidence-backed fine-tuning planner."
}
```

Requirements:

- both fields are strings;
- `completion` is non-empty after trimming;
- `prompt` can be empty.

Runtime behavior:

1. encode the prompt with special tokens;
2. encode completion without added special tokens;
3. append the tokenizer EOS token when available and absent;
4. retain completion tokens first, up to `sequence_length`;
5. use remaining capacity for the rightmost prompt tokens; and
6. mask prompt labels with `-100`.

At least one supervised completion token must remain.

## Instruction-output schema

```json
{
  "instruction": "Summarize the passage.",
  "input": "Aptus compares bounded fine-tuning strategies.",
  "output": "Aptus compares fine-tuning plans."
}
```

Requirements:

- `instruction` and `output` are strings;
- `output` is non-empty after trimming;
- `input` is optional and included only when it is a non-empty string.

The runtime constructs this prompt exactly:

```text
### Instruction:
<trimmed instruction>
### Input:
<trimmed optional input>
### Response:
```

`output` is the supervised completion. Truncation and masking follow the same
completion-first rule as prompt-completion rows.

## Messages schema

```json
{
  "messages": [
    {"role": "system", "content": "Answer precisely."},
    {"role": "user", "content": "What does Aptus emit?"},
    {"role": "assistant", "content": "A validated training bundle."}
  ]
}
```

Requirements:

- `messages` is a non-empty list;
- every item is an object with string `role` and `content`;
- the final item has role `assistant`;
- the final assistant content is non-empty.

Runtime behavior uses the pinned tokenizer's chat template twice:

1. render all turns except the final assistant with
   `add_generation_prompt=true`;
2. render the full conversation with `add_generation_prompt=false`;
3. require the prompt token sequence to be an exact prefix of the full token
   sequence;
4. supervise only the final assistant suffix; and
5. preserve completion tokens first and left-truncate the prompt suffix.

If the pinned chat template is not prefix-separable, Aptus fails closed instead
of rewriting its control-token format.

## Extra fields and metadata

Canonical compilation serializes the complete original object with sorted keys.
Extra fields are retained. They are not passed to the model unless they are part
of the selected schema transformation.

This permits provenance and review metadata such as:

```json
{
  "instruction": "State the conclusion.",
  "output": "The conclusion.",
  "metadata": {
    "source_id": "work-0042",
    "review_status": "approved",
    "split_group": "work-0042"
  }
}
```

Aptus preserves those values but does not validate reviewer identity, consent,
source rights, PII, rubric content, or approval workflow.

## Empty and malformed rows

The profiler and compiler ignore these supported but empty shapes:

- empty or whitespace-only `text` or `content`;
- prompt-completion rows without a non-empty completion;
- instruction-output rows without a non-empty output; and
- an empty `messages` list.

They increment `empty_count` and do not enter `data/training.jsonl`.

Malformed structured rows fail the dataset. Examples include a non-object
message, non-string message content, a non-assistant final turn, or an object
with no supported schema. The error includes source-row context during
profiling or compilation.

## Profiling behavior

Profiling always reads the complete file to calculate:

- source SHA-256 and source byte size;
- valid and empty row counts;
- schema counts and `mixed` classification;
- total token estimate or tokenizer measurement;
- canonical and maximum-row byte estimates;
- normalized duplicate count; and
- truncation count when sequence length is supplied.

Without a tokenizer, token count is `ceil(character_count / 4)` with a minimum
of one. This is labeled `estimated`. Duplicate detection collapses whitespace
for hashing and emits a warning only. It does not remove duplicates.

When a sample limit is set, a digest-seeded deterministic reservoir supplies
the p50, p95, maximum, and sample indices. Totals and schema checks still cover
every row.

## Canonical compilation

Compilation:

1. copies the source bytes to `data/dataset.<suffix>`;
2. verifies the copy against the profiled SHA-256;
3. repeats schema validation over every row; and
4. writes every valid non-empty object to `data/training.jsonl` with UTF-8,
   sorted keys, non-ASCII preservation, and one newline per object.

The bundle plan rewrites `source_path` to the copied relative path and records
`bundle_path`. A source change after profiling aborts compilation.

## Pilot pressure set

The compiler selects up to the required row count from the longest extracted
text values, with stable source order as the tie-break. Required count is:

```text
max(32, 2 * effective_batch_size)
```

When the source has fewer rows, Aptus repeats real rows deterministically until
the count is reached. It writes the result to `data/pilot-sample.jsonl`.

Pilot training shuffles this bounded set with seed 17, disables evaluation, and
pads each batch to the configured sequence length. This is pressure evidence,
not a representative evaluation sample.

## Full-run split groups

A row can declare a group in either location:

```json
{"split_group": "document-17", "text": "..."}
```

```json
{
  "text": "...",
  "metadata": {"split_group": "document-17"}
}
```

Rules:

- the value must be a non-empty string;
- when both locations exist, their strings must be exactly equal;
- every row with the same value is one atomic split unit;
- each ungrouped row is its own split unit; and
- raw group names are not emitted in public split evidence.

Aptus does not infer groups. Corpus authors must group every unit that would
cause leakage if split across training and evaluation.

## Deterministic split algorithm

For `N` rows and requested fraction `f`, the target evaluation count is:

```text
min(N - 1, max(1, round(N * f)))
```

when `f > 0` and more than one split unit exists. Otherwise the target is zero.

Ungrouped data uses `deterministic-exact-row-count-sha256`. Grouped data uses
`deterministic-size-aware-group-sha256`:

1. group counts are collected;
2. deterministic SHA-256 priorities order units;
3. exact subset selection finds a group row total that can be completed by
   available ungrouped rows;
4. if exact target is impossible, the closest feasible group total is selected
   with deterministic tie-breaking; and
5. ungrouped rows fill the remaining target where possible.

At least one training row is retained. An indivisible group can make the
realized fraction differ from the requested value.

## Split evidence and mutation checks

`aptus.dataset-split.v1` records:

- strategy and seed;
- requested fraction, target row count, realized fraction, and row error;
- total, training, and evaluation row counts;
- declared-group counts by side;
- ungrouped-row and split-unit counts;
- canonical JSONL SHA-256; and
- assignment SHA-256.

The canonical file is hashed during all three split passes. A change aborts the
run. Lazy datasets open and hash the file descriptor, verify file identity, and
recheck stability while reading. Distributed ranks must agree on canonical
digest, assignment digest, and counts before training continues.

## Related documentation

- [Plan schema](plan-schema.md)
- [Bundle manifest](bundle-manifest.md)
- [Evidence records](evidence-records.md)
- [Reviewed corpus contract](reviewed-corpus-contract.md)
- [Model, dataset, and hardware facts](../guides/model-dataset-hardware.md)
- [Capability matrix](capability-matrix.md)
