# Aptus ↔ Veriformis handoff

> **Status:** Active | **Authority:** Normative Aptus consumer contract | **Applies to:** Aptus 0.2 intake from a dataset compiler | **Audience:** Dataset authors and Aptus operators | **Last reviewed:** 2026-08-21 | **Review by:** When Aptus grows a `.vfbundle` importer or Veriformis ships a public beta

Veriformis is a separate product. It is not done. Aptus does not ingest raw
documents and does not import a `.vfbundle` today. The handoff is **local
JSONL**. This page is what Aptus will accept. It is not a Veriformis ship
claim.

A Veriformis-side descriptor exists (`veriformis.aptus-handoff/v1`). Aptus 0.2
does not read that file. Do not treat a sibling `.aptus-handoff.json` as an
Aptus compile input.

## Who owns what

| Concern | Owner | Aptus will not |
|---|---|---|
| Ingest, clean, chunk, curate, seal, provenance | Veriformis | Parse PDFs, encyclicals, or HTML |
| Name gold / recitation rows | Dataset author (Veriformis when it can) | Invent gold from the training file |
| CUDA train/eval assignment | Aptus (`split_group` or ungrouped hash/seed) | Recompute Veriformis leakage groups; honor `--include` on CUDA |
| MLX compiled valid tail | Aptus (last `round(n * evaluation_fraction)` rows) | Treat Veriformis `evaluation.jsonl` as that tail |
| Recitation exact-match | Aptus `eval` / `eval-generate` | Call that score quality |

## Three files, three jobs

Give Aptus these paths. Do not concatenate them.

| File | Role | Aptus command |
|---|---|---|
| `corpus.jsonl` | Every SFT row Aptus may train on | `profile`, `prepare-train --corpus`, `emit-run --dataset` |
| `gold.jsonl` | Named recitation rows; **must already be in corpus** | MLX: `prepare-train --include`. CUDA and scoring: `eval --gold` |
| `holdout.jsonl` | Generalization holdout (Veriformis `data/evaluation.jsonl`) | Optional later eval. **Never** the MLX train-file tail |

Veriformis names: `data/train.jsonl` → Aptus `corpus.jsonl`;
`data/evaluation.jsonl` → Aptus `holdout.jsonl`. Recitation gold is **not**
that evaluation partition. Concatenating holdout onto the end of the MLX
training file parks those rows in valid. Exact-match on them is then 0 even
when train recites.

## Row schema Aptus consumes

Preferred for MLX QLoRA:

```json
{"id": "row-1", "prompt": "…", "completion": "…", "split_group": "document-a"}
```

Rules Aptus already enforces:

- JSONL, one object per non-blank line.
- `prompt` and `completion` are non-empty strings (`prepare-train` refuses
  otherwise). Duplicate prompts are refused.
- Optional `id` for eval identity. Optional `split_group` for CUDA grouped
  splits (must match Veriformis leakage grouping when both are used).
- MLX compile still rejects whole-text `{"text": "…"}` rows.

Instruction/output and messages rows remain legal for `profile`/`compile`.
`prepare-train --include` is prompt/completion only.

## Split ownership

1. **Veriformis** assigns sealed train vs evaluation partitions (no leakage
   across groups). Recitation gold stays in the train partition.
2. **Aptus CUDA** may split that training file again by `split_group` or
   ungrouped hash/seed. `emit-run --include` is refused on CUDA. Recitation
   gold must share the source document's `split_group`; a gold-only group can
   hash entirely into CUDA eval, and exact-match is then 0 even if the rest of
   the document was trained.
3. **Aptus MLX** always takes the last 10% (default) of the **compiled training
   file** as valid. `aptus prepare-train --include` keeps named gold in the
   train prefix and refuses gold prompts that are not already in the corpus.

Do not ask Aptus to honor Veriformis `evaluation.jsonl` as the MLX valid split.
Do not ask Veriformis to emit MLX `data/mlx/valid.jsonl`. That file is compiler
output.

## What Aptus will not do in 0.2

- Import `.vfbundle` or `veriformis.aptus-handoff/v1`.
- Train on provenance JSONL.
- Infer gold from holdout.
- Merge Veriformis into Aptus.

When Veriformis is ready, the operator copies JSONL out of the sealed bundle
and runs `prepare-train` then `emit-run`. Full train still needs
`--confirm-full-train`.

## Related documentation

- [Prepare a dataset](prepare-a-dataset.md)
- [Dataset schemas](../reference/dataset-schemas.md)
- [CLI `prepare-train`](../reference/cli.md#aptus-prepare-train)
- [CLI `emit-run`](../reference/cli.md#aptus-emit-run)
- [Design an evaluation](design-an-evaluation.md)
