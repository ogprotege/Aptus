# Operator MLX recipe helpers

Prefer the product CLI:

```bash
aptus prepare-train --corpus CORPUS.jsonl --include GOLD.jsonl --output TRAIN.jsonl
aptus emit-run FACTS --dataset TRAIN.jsonl --output RUNDIR --objective quality --compile
```

These `tools/operator_mlx/` scripts remain compatibility wrappers. Aptus plans,
compiles, validates, and locally executes. The wrappers do the
operator work that was missing around Journey B:

1. Put recitation rows in the **compiled train prefix** (MLX valid is the last
   10% of the file).
2. Emit `spec-plan` from **live hardware** with the measured pin (`--objective
   quality` → rank 16). The CLI default `--objective memory` → rank 8 did not
   recit this corpus.
3. Generate CompletionsDataset-shaped predictions and score them with
   `aptus eval`.

They do **not** replace `--confirm-full-train`. They do **not** pick an
optimum. They encode what was measured on this M5 Pro, 2026-08-20.

## Measured recipe (this host, this 7B pin)

| Knob | Emit this | Do not emit |
| --- | --- | --- |
| Objective | `quality` | `memory` (rank 8; 0/62) |
| Rank | enumerator 16 | hand-set 32/64 |
| Epochs | **5** | **10** (loss 0.05 → 7) |
| Gold in file | mixed into train prefix | concatenated at the end; gold-block prefix |
| 7B 28-layer | `--confirm-unreviewed-runtime` | claiming Path Alpha |
| Eval | greedy CompletionsDataset; field `prediction` | threshold 1.0 as a quality yes |

Evidence: `aptus-work/magisterium-mix/` job `job_1bfe7e9c7fbf4600b332849377eed547`,
original 62-row gold exact-match **38/62**. Not a reviewed 7B. Not quality.

## Run (from repo root, venv on)

Prepare a train JSONL that keeps gold out of the MLX valid tail:

```bash
python tools/operator_mlx/prepare_sft_train_file.py \
  --corpus aptus-work/magisterium-b2/corpus.jsonl \
  --include aptus-work/magisterium/gold.jsonl \
  --output aptus-work/operator-run/train.jsonl \
  --manifest aptus-work/operator-run/split-manifest.json
```

Probe hardware and write `spec-plan.sh` (optional: actually plan + compile):

```bash
PYTHONPATH=src:. python tools/operator_mlx/emit_spec_plan.py \
  --dataset aptus-work/operator-run/train.jsonl \
  --workdir aptus-work/operator-run \
  --run-plan --compile
```

Then `./aptus-work/operator-run/ladder.sh`. Full train is printed, not run,
until you pass `--confirm-full-train` yourself.

After a train, score gold against the adapter:

```bash
PYTHONPATH=src:. python tools/operator_mlx/eval_mlx_adapter.py \
  --gold aptus-work/magisterium/gold.jsonl \
  --adapter aptus-work/operator-run/bundle/runs/run_<id>/final \
  --workdir aptus-work/operator-run/eval
```
