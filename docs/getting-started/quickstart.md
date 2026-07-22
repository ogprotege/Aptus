# CUDA Target-Host Quickstart

> **Status:** Active | **Authority:** Operational tutorial | **Applies to:** Aptus 0.2 | **Audience:** CUDA operators | **Last reviewed:** 2026-07-22 | **Review by:** 2026-10-22 or when the runtime sequence changes

This guide is a target-host template. It creates a plan, compiles it, and
follows the supported CUDA runtime order. Every uppercase model value and every
numeric hardware value must be replaced with facts for the intended run.

For a copy-and-paste workflow that downloads no model and starts no training,
use the [first planning-only run](first-plan.md).

## Before you start

Confirm all of the following:

- the model repository is pinned to a real 40-to-64-character hexadecimal
  commit;
- you reviewed the model license and can attest training permission;
- the dataset is authorized, reviewed, and backed up;
- the host has supported CUDA hardware and enough free disk;
- package and model downloads are acceptable on this machine; and
- the bundle and state paths are new or intentionally selected.

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
  --revision REAL_40_TO_64_HEX_COMMIT
```

Inspection does not grant training permission. Verify the license and confirm
training permission yourself.

## 3. Create a plan

```bash
aptus spec-plan \
  --model-id provider/model \
  --revision REAL_40_TO_64_HEX_COMMIT \
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

## 6. Prepare the target runtime environment

Managed jobs execute with the interpreter that launches `aptus`. Create an
external environment containing both Aptus and the selected bundle stack:

```bash
BUNDLE_DIR="$(pwd)/aptus-work/bundle"
python -m venv ./aptus-work/runtime-env
source ./aptus-work/runtime-env/bin/activate
python -m pip install -e .
python -m pip install -r "$BUNDLE_DIR/requirements.txt"
python -c "import aptus, torch, transformers; print('runtime imports passed')"
```

The environment is beside the sealed bundle, not inside it. If this host uses
an installed Aptus wheel rather than a repository checkout, install that wheel
instead of the editable package. Do not launch managed jobs from an interpreter
that lacks the generated requirements.

## 7. Run the five runtime actions

Submit one managed job at a time and wait for successful completion:

```bash
aptus run "$BUNDLE_DIR" --action dependency
aptus run "$BUNDLE_DIR" --action model-data
aptus run "$BUNDLE_DIR" --action preflight
aptus run "$BUNDLE_DIR" --action pilot
aptus run "$BUNDLE_DIR" --action train --confirm-full-train
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

## 8. Read the result

A managed full run writes to `$BUNDLE_DIR/runs/run_*`. Aptus never reuses that path.
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

## Related documentation

- [Choose your path](choose-your-path.md)
- [Operator checklist](../operations/operator-checklist.md)
- [Configuration and defaults](../reference/configuration-defaults.md)
- [Troubleshooting](../guides/troubleshooting.md)
- [Release gates](../operations/release-gates.md)
