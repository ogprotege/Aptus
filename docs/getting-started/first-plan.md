# First Planning-Only Run

> **Status:** Active | **Authority:** Operational tutorial | **Applies to:** Aptus 0.2 | **Audience:** First-time users | **Last reviewed:** 2026-07-22 | **Review by:** 2026-10-22 or when CLI defaults change

This tutorial proves the local planning, compilation, and static-validation
path. It works without CUDA. It uses bundled synthetic data and synthetic model
facts. It does not download a model or start training.

## Expected result

You will create:

```text
aptus-work/
  plan.json
  bundle.zip
  bundle/
    bundle-manifest.json
    plan.json
    requirements.txt
    runbook.md
    validate.py
    ...
```

The final command should return a report with `"state": "static-pass"`.
That result proves the bundle contract and generated source passed local static
checks. It does not prove dependency installation, model compatibility, CUDA
fit, training completion, or model quality.

## 1. Install the development package

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[server,test]'
aptus --help
```

## 2. Profile the bundled example

```bash
mkdir -p ./aptus-work

aptus profile \
  --dataset ./examples/support-sft.jsonl \
  --sequence-length 128 \
  --output ./aptus-work/dataset-profile.json
```

The profile should report four valid rows, a measured source digest, and sampled
length statistics. The data is synthetic and is not a quality benchmark.

## 3. Create a real Aptus plan

The revision below is valid contract syntax but does not identify a real model.
That is acceptable for this planning-only exercise because `spec-plan` does not
load model weights.

```bash
aptus spec-plan \
  --model-id example/model \
  --revision 0123456789abcdef0123456789abcdef01234567 \
  --family llama \
  --parameters-b 0.01 \
  --hidden-size 128 \
  --intermediate-size 256 \
  --layers 2 \
  --context-length 512 \
  --license tutorial-only \
  --confirm-training-allowed \
  --dataset ./examples/support-sft.jsonl \
  --backend cuda \
  --gpu-count 1 \
  --vram-gib 24 \
  --free-vram-gib 22 \
  --bf16 \
  --four-bit \
  --eight-bit \
  --host-ram-gib 64 \
  --host-ram-free-gib 48 \
  --reserve-gib 2 \
  --disk-free-gib 200 \
  --objective memory \
  --sequence-length 128 \
  --effective-batch-size 4 \
  --epochs 1 \
  --evaluation-fraction 0.25 \
  --checkpoint-steps 10 \
  --output ./aptus-work/plan.json
```

Open `aptus-work/plan.json` and confirm:

- `schema_version` is `aptus.training-plan.v3`;
- `formula_version` is `aptus-memory-v2`;
- twelve candidates are present;
- each candidate retains a status and reason;
- `recommended` names one viable candidate; and
- every point and upper memory component is visible.

The entered CUDA values are tutorial planning facts. They are not measurements
of this Mac and cannot authorize execution.

## 4. Compile the selected candidate

```bash
aptus compile \
  --plan ./aptus-work/plan.json \
  --output ./aptus-work/bundle
```

Compilation creates `aptus-work/bundle` and `aptus-work/bundle.zip`. It refuses
to overwrite either. Use a new path when repeating the tutorial.

## 5. Validate the bundle statically

```bash
aptus validate ./aptus-work/bundle --level static
```

Review the returned `checked_files`, `bindings`, and `findings`. A valid result
has no findings and reports `static-pass`.

## 6. Stop at the evidence boundary

Do not run dependency, model-data, preflight, pilot, or train actions with these
synthetic model facts. Those actions require a real provider model, an immutable
revision, verified training rights, compatible dependencies, and supported CUDA
hardware.

For a real run, continue with the [quickstart](quickstart.md), replace every
fact, and follow the [operator checklist](../operations/operator-checklist.md).

## Common rerun issue

If compilation reports that output already exists, this is expected no-clobber
behavior. Choose a new directory and archive name. Do not patch or clear a
compiled bundle in place.

## Related documentation

- [Choose your path](choose-your-path.md)
- [Install Aptus](install.md)
- [CLI reference](../reference/cli.md)
- [Bundle manifest](../reference/bundle-manifest.md)
- [Validation states](../reference/validation-states.md)
