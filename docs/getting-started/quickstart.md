# Quickstart

This guide creates a plan, compiles it, and follows the supported runtime order.
Use real facts for a real run. The values below are examples only.

## 1. Prepare data

Aptus accepts local `.jsonl`, `.json`, `.csv`, and `.txt` files. The compiler
copies the source, validates every supported source-schema row, and writes each
one deterministically to `data/training.jsonl`. Tokenizer-specific input and
loss-mask transformation occurs during model-data validation and training.

```bash
aptus profile \
  --dataset examples/support-sft.jsonl \
  --sequence-length 1024 \
  --output ./aptus-work/dataset-profile.json
```

## 2. Inspect optional facts

Inspect hardware on the server host:

```bash
aptus inspect hardware
```

CUDA hosts report visible devices and current capacity. Darwin arm64 hosts
without CUDA report an `mps` discovery record for shared unified memory. That
record is inventory only. It does not make an MPS or MLX training candidate
executable, because v0.2 execution remains CUDA-only.

Inspect bounded model metadata for a pinned repository revision:

```bash
aptus inspect model \
  --model-id provider/model \
  --revision IMMUTABLE_COMMIT
```

Inspection does not grant training permission. Verify the license and confirm
training permission yourself.

## 3. Create a plan

```bash
aptus spec-plan \
  --model-id provider/model \
  --revision IMMUTABLE_COMMIT \
  --family llama \
  --parameters-b 7 \
  --hidden-size 4096 \
  --intermediate-size 11008 \
  --layers 32 \
  --context-length 4096 \
  --license LICENSE_NAME \
  --confirm-training-allowed \
  --dataset examples/support-sft.jsonl \
  --sample-limit 512 \
  --backend cuda \
  --gpu-count 1 \
  --vram-gib 24 \
  --free-vram-gib 22 \
  --bf16 --four-bit --eight-bit \
  --host-ram-gib 64 \
  --host-ram-free-gib 48 \
  --reserve-gib 2 \
  --disk-free-gib 200 \
  --objective memory \
  --sequence-length 1024 \
  --effective-batch-size 16 \
  --epochs 1 \
  --evaluation-fraction 0.1 \
  --checkpoint-steps 100 \
  --output ./aptus-work/plan.json
```

Read the candidate list, unsupported reasons, assumptions, and evidence records.
The recommended candidate is only the highest-ranked viable member of the
enumerated catalog. Viable includes `feasible` and `conditional`, with feasible
ranked first. A conditional recommendation is not a measured fit.

## 4. Compile

```bash
aptus compile \
  --plan ./aptus-work/plan.json \
  --output ./aptus-work/bundle
```

Compilation refuses a non-empty output directory. It also creates a deterministic
ZIP beside the bundle and refuses to overwrite an existing archive.

## 5. Run static validation

```bash
aptus validate ./aptus-work/bundle --level static
```

Static validation checks contracts, identities, paths, generated source, direct
pins, hashes, and manifest coverage. It does not import the training stack or
allocate CUDA memory.

## 6. Run the five runtime actions

Submit one managed job at a time and wait for successful completion:

```bash
aptus run ./aptus-work/bundle --action dependency
aptus run ./aptus-work/bundle --action model-data
aptus run ./aptus-work/bundle --action preflight
aptus run ./aptus-work/bundle --action pilot
aptus run ./aptus-work/bundle --action train --confirm-full-train
```

Use these commands between actions:

```bash
aptus jobs
aptus jobs --id JOB_ID
```

The model-data action resolves the pinned model and tokenizer, validates the
loaded parameter count and plan-driving structural config facts against the
plan, prepares the selected method, enforces its trainable-parameter scope, and
transforms every canonical row. The preflight repeats the method-scope census
with synthetic tensors. The pilot uses a bounded real model and data pressure
set in two fresh processes, requires the same census in both phases, and
confirms checkpoint continuation.

Train admission performs a deep, atomic recheck of pilot bindings and current
VRAM, host RAM, disk, environment, and artifacts. A prior status display is not
the authority for admission.

## 7. Read the result

A managed full run writes to `bundle/runs/run_*`. Aptus never reuses that path.
After the child process exits successfully, the parent verifies metrics and the
structural safetensors export file tree. Only then does it mark the job complete
and promote `validation-report.json` to `measured-run-pass`.

For a full run, the generated trainer deterministically assigns canonical rows
to train and evaluation. Declared `split_group` values remain whole. Metrics bind
the canonical JSONL digest, assignment digest, target and realized evaluation
sizes, and trainable-parameter census. Large indivisible groups can make the
realized evaluation fraction differ from the requested fraction, so read the
recorded error rather than assuming an exact ratio.

No real CUDA pilot has been completed on the current development Mac. Complete
the release gates before treating this path as release-ready.
