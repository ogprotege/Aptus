# Aptus
An evidence-backed fine-tuning planner and artifact compiler. Give Aptus explicit
model, dataset, hardware, and target facts; it compares feasible strategies,
explains its assumptions and tradeoffs, and emits a validated training bundle.

## Project status

Aptus is a greenfield rebuild. The historical `HyperTune/` working folder was
audited, its useful concepts were accounted for, and the folder was removed.
The user retains a separate local backup; no legacy runtime code is part of
Aptus.

The forensic recovery audit is available at
[`docs/audits/aptus-legacy/`](docs/audits/aptus-legacy/). Start with the
[`executive summary`](docs/audits/aptus-legacy/executive-summary.md), then review
the [`hidden gems`](docs/audits/aptus-legacy/hidden-gems.md) and
[`architecture options`](docs/audits/aptus-legacy/architecture-options.md).

## Current vertical slice

The first working slice supports:

- deterministic local JSON, JSONL, CSV, and text profiling;
- explicit open-model and per-device CUDA hardware facts;
- LoRA and QLoRA feasibility comparison;
- component-level `heuristic-v1` memory estimates with a safety margin;
- quality, memory, and speed objectives;
- a version-pinned Transformers/PEFT bundle;
- dependency-free validation and a fully offline one-step adapter smoke test.

The memory model is transparent but not yet calibrated. Aptus does not describe
its output as universally optimal or guaranteed to fit.

## Example

```bash
PYTHONPATH=src python3 -m aptus plan \
  --model-id meta-llama/example-7b \
  --revision aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --family llama \
  --parameters-b 7 \
  --hidden-size 4096 \
  --layers 32 \
  --context-length 4096 \
  --license example \
  --confirm-training-allowed \
  --dataset tests/fixtures/text.jsonl \
  --backend cuda \
  --gpu-count 1 \
  --vram-gib 24 \
  --bf16 \
  --four-bit \
  --host-ram-gib 64 \
  --reserve-gib 2 \
  --objective quality \
  --sequence-length 128 \
  --effective-batch-size 8 \
  --epochs 1 \
  --output ./aptus-bundle
```

Then inspect `plan.json` and `validation-report.json` before running:

```bash
python aptus-bundle/validate.py
python aptus-bundle/train.py --smoke
python aptus-bundle/train.py
```

Design boundaries and acceptance criteria are documented in
[`docs/design/aptus-core-vertical-slice.md`](docs/design/aptus-core-vertical-slice.md).
The successful pinned offline smoke is documented in
[`docs/validation/aptus-core-smoke.md`](docs/validation/aptus-core-smoke.md).
