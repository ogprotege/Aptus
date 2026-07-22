# Reviewed corpus contract

Status: accepted intake contract for supervised fine-tuning rows. Aptus does
not yet provide a chat-capture service or reviewer application.

The historical `EXAMPLE/Chainlit_05-03-25` application had the right product
instinct: capture difficult interactions, let a human correct them, then use
the approved material to improve a model. Its implementation did not preserve
the governance needed to make that safe or reproducible. Raw chat history,
thumbs-up events, or generated answers are not training data merely because
they were collected.

## Boundary

A row may enter an Aptus training bundle only after an external intake process
has:

1. assigned immutable interaction, turn, and record identifiers;
2. recorded the model ID, immutable model revision, generation settings, and
   system-prompt digest that produced the candidate answer;
3. recorded source citations and the right to use each source;
4. applied the project's consent, retention, and purpose policy;
5. removed or approved personally identifying and confidential material;
6. attached a human correction or explicit approval;
7. recorded reviewer identity, rubric version, and review time;
8. deduplicated the content and assigned a leakage-prevention group; and
9. exported only the approved supervised target.

The capture system remains outside the training trust boundary. Aptus hashes
the exported file, preserves every extra metadata field in canonical JSONL,
and trains only on the supported text, completion, instruction, or chat fields.

## Canonical reviewed row

```json
{
  "messages": [
    {"role": "user", "content": "A reviewed question"},
    {"role": "assistant", "content": "The human-approved answer"}
  ],
  "split_group": "source:work-42",
  "metadata": {
    "record_id": "rec_01JEXAMPLE",
    "interaction_id": "int_01JEXAMPLE",
    "turn_id": "turn_01JEXAMPLE",
    "review": {
      "status": "approved",
      "reviewer_id": "reviewer:pseudonymous-id",
      "rubric_id": "rubric:domain-sft-v1",
      "reviewed_at": "2026-07-22T00:00:00Z"
    },
    "provenance": {
      "source_ids": ["source:work-42#section-8"],
      "license_or_permission": "project-attested",
      "consent_status": "approved-for-training",
      "pii_review": "passed"
    },
    "generation": {
      "model_id": "provider/model",
      "revision": "immutable-provider-commit",
      "system_prompt_sha256": "64-lowercase-hex-characters",
      "parameters": {"temperature": 0.2, "top_p": 0.9}
    }
  }
}
```

`messages`, `prompt` plus `completion`, `instruction` plus `output`, and plain
`text` remain the supported SFT payloads. Metadata never replaces the
supervised target.

## Leakage-prevention groups

`split_group` is an optional, explicit top-level string. Related chunks,
paraphrases, questions, or turns from the same source unit must use the same
value. The generated trainer deterministically assigns an entire group to
training or evaluation. It never places rows with the same declared group in
both partitions. Grouped data uses a size-aware assignment, so a large
indivisible group can make the realized evaluation fraction differ from the
requested fraction.

Good group identities describe the source unit that could leak:

- one book, article, case, policy, or document;
- one original conversation or support incident;
- one generated seed before paraphrase expansion; or
- one question family when variants share the answer.

Do not use a unique row ID as `split_group`. That defeats the boundary. Rows
without a group remain independently assigned, and the split evidence records
how many rows were ungrouped. A target-quality test set remains a separate,
immutable evaluation artifact and must not be included in the training file.

## Review states

An intake service should use an append-only state history:

```text
captured -> redacted -> reviewed -> approved -> exported
                         |             |
                         +-> rejected  +-> revoked
```

Only `approved` records may be exported. A correction creates a new revision
linked to the prior record. It never overwrites the raw event. Revocation
prevents future exports and records which prior dataset digests contained the
record.

## Required rejection gates

The exporter must reject a row when:

- the assistant target is empty or consists only of copied prompt text;
- consent, license, source, reviewer, or rubric facts are missing;
- PII or confidential-data review has not passed;
- the reviewer approved only a rating but not the actual target text;
- the model answer is exported instead of the reviewed correction;
- record, interaction, or turn identity collides;
- provenance points to a mutable model or source revision;
- the same normalized content already exists under conflicting labels; or
- the split group is missing when rows are known to share a source.

## Current and future Aptus behavior

Current Aptus preserves these metadata fields, hashes the complete canonical
dataset, and masks prompt tokens for supervised completion rows. During a full
run, it respects declared split groups, detects canonical-data changes during
split and lazy consumption, requires distributed ranks to agree on split
bindings, and records target and realized split evidence without group names.
It also rejects zero, non-finite, or method-scope-invalid trainable parameters.

Aptus does not determine whether a declared group is the correct leakage unit.
It does not verify consent, reviewer signatures, rubric contents, source
licenses, PII decisions, or revocation records. Those checks belong to a future
typed corpus-ingestion service and must not be implied by a successful training
bundle.
