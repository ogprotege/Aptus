# Path Beta operator runbook — CUDA LoRA handoff

> **Status:** Active | **Authority:** Path-scoped operator procedure | **Applies to:** Aptus 0.2 Path Beta (`path-beta-cuda-lora-single-v1`) | **Audience:** Solo operators with a Mac (or any) control plane and one CUDA host | **Last reviewed:** 2026-08-12 | **Review by:** When Path Beta identity or CUDA pins change

This runbook completes **Journey B** for the frozen Path Beta identity: plan and
compile on a control machine, hand off a sealed bundle to one CUDA host, install
a **clean** environment, and finish the ordered ladder through
`measured-run-pass`.

It is **not** general CUDA certification, multi-GPU proof, or model quality.

## Hardware prerequisites (CUDA host)

- Ubuntu 24.04.x preferred (historical/campaign class: **24.04.4 LTS**)
- NVIDIA GeForce **RTX 3050** class (~8 GiB VRAM) preferred; other cards require
  an explicit host-class change decision
- Driver with CUDA support sufficient for the bundle Torch pin (acceptance used
  driver **595.84**, Torch **2.13.0+cu130**)
- Tens of GiB free disk for HF cache, venv, and run outputs
- Hugging Face access for the pinned model revision (or a pre-populated cache)

Control plane may be a Mac without CUDA. Declare CUDA hardware facts from
measured host values; do **not** imply Mac runs CUDA train.

## Install Aptus (control plane)

```bash
cd /path/to/Aptus
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[server,test]'
export PYTHONPATH=src:.
```

## Exact Path Beta identity

| Field | Value |
| --- | --- |
| Path ID | `path-beta-cuda-lora-single-v1` |
| Model | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| Revision | `12fd25f77366fa6b3b4b768ec3050bf629380bac` |
| Family / architecture | `llama` / `LlamaForCausalLM` |
| Parameters | 134,515,008 (`--parameters-b 0.134515008`) |
| Hidden / intermediate / layers | 576 / 1536 / 30 |
| Context | 8192 |
| Dataset | `examples/support-sft.jsonl` |
| Dataset SHA-256 | `bf2dca3d6398d639f47a883203920e1f52b0981becac96734147054e53f8aa44` |
| Method / placement | LoRA BF16 / `single` |
| Runtime | `transformers-peft-cuda` on `cuda` |

Verify dataset digest:

```bash
shasum -a 256 examples/support-sft.jsonl
```

## Measure host facts (on CUDA host)

```bash
python - <<'PY'
import torch
props = torch.cuda.get_device_properties(0)
free, total = torch.cuda.mem_get_info(0)
print("total_vram_bytes", total)
print("free_vram_bytes", free)
print("total_gib", total / 1024**3)
print("free_gib", free / 1024**3)
print("name", props.name)
PY
```

Use **exact** Torch-visible totals when planning. A slightly inflated
`--vram-gib` fails closed at model-data hardware parity.

## Plan (control plane)

```bash
WORKDIR=./aptus-work/path-beta
mkdir -p "$WORKDIR"

# Replace with measured host values
VRAM_GIB=7.656005859375
FREE_VRAM_GIB=7.234619140625
HOST_RAM_GIB=62.6
HOST_FREE_GIB=58
DISK_GIB=189

python -m aptus spec-plan \
  --model-id HuggingFaceTB/SmolLM2-135M-Instruct \
  --revision 12fd25f77366fa6b3b4b768ec3050bf629380bac \
  --family llama --parameters-b 0.134515008 \
  --model-type llama --architecture LlamaForCausalLM \
  --hidden-size 576 --intermediate-size 1536 --layers 30 \
  --context-length 8192 --license apache-2.0 --confirm-training-allowed \
  --dataset ./examples/support-sft.jsonl --sample-limit 64 \
  --backend cuda --training-runtime transformers-peft-cuda --gpu-count 1 \
  --vram-gib "$VRAM_GIB" --free-vram-gib "$FREE_VRAM_GIB" --bf16 \
  --host-ram-gib "$HOST_RAM_GIB" --host-ram-free-gib "$HOST_FREE_GIB" \
  --reserve-gib 2 --disk-free-gib "$DISK_GIB" \
  --objective speed --sequence-length 128 --effective-batch-size 1 \
  --epochs 1 --prefer-method lora --evaluation-fraction 0.25 \
  --checkpoint-steps 1 --optimizer-steps 3 \
  --output "$WORKDIR/plan.json"
```

Expect **recommended** `lora` / `single` / **feasible** (or conditional when
pilot-required labels apply). Stderr prints refusal guidance for non-viable rows
(M2). Plan schema is `aptus.training-plan.v6`.

## Compile and static-validate (control plane)

```bash
python -m aptus compile \
  --plan "$WORKDIR/plan.json" \
  --output "$WORKDIR/bundle"

python -m aptus validate "$WORKDIR/bundle" --level static
```

Compilation refuses a non-empty output directory and writes
`$WORKDIR/bundle.zip`. Record the **artifact fingerprint** from the validation
report.

## Handoff integrity checklist

Copy to the CUDA host (example layout):

```text
~/aptus-m4/Aptus/          # source checkout at measured commit (for managed jobs)
~/aptus-m4/path-beta-m4/   # plan.json, bundle/, bundle.zip
```

On the host:

1. Verify ZIP or bundle fingerprint matches the control-plane compile.
2. Confirm `requirements.txt` is the sealed direct pin set (do not hand-edit).
3. Confirm `policy/model-policy-snapshot.v1.json` is present.
4. Do **not** create a venv inside the sealed bundle directory.

## Clean host environment

```bash
cd ~/aptus-m4
/usr/bin/python3 -m venv runtime-env
source runtime-env/bin/activate
python -m pip install -U pip wheel

# Install CUDA Torch matching the public pin (local label OK)
python -m pip install 'torch==2.13.0' --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r path-beta-m4/bundle/requirements.txt
python -m pip install -e './Aptus'

python - <<'PY'
import torch, transformers, peft, aptus
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
print(transformers.__version__, peft.__version__, aptus.__version__)
PY
```

Dependency gate compares **public** versions, so `2.13.0+cu130` satisfies
`torch==2.13.0`. A wrong public release still fails closed.

## Ordered managed ladder (CUDA host)

```bash
cd ~/aptus-m4
source runtime-env/bin/activate
export PYTHONPATH=~/aptus-m4/Aptus/src
BUNDLE=~/aptus-m4/path-beta-m4/bundle
STATE=~/aptus-m4/state
mkdir -p "$STATE"

python -m aptus run --action dependency --state-dir "$STATE" "$BUNDLE"
python -m aptus run --action model-data --state-dir "$STATE" "$BUNDLE"
python -m aptus run --action preflight --state-dir "$STATE" "$BUNDLE"
python -m aptus run --action pilot --state-dir "$STATE" "$BUNDLE"
python -m aptus run --action train --confirm-full-train --state-dir "$STATE" "$BUNDLE"
```

Stop at the first failed gate. Cost control: the synthetic Path Beta ladder is
short (minutes), but HF model download and CUDA stack install dominate first-run
time.

### How to read pass vs refuse

| Signal | Meaning |
| --- | --- |
| Job `state: completed`, `return_code: 0` | Gate passed |
| Job `state: failed` + log RuntimeError | Gate refused; fix facts/env and re-run with fresh state when required |
| Bundle `validation-report.json` `state: measured-run-pass` | Full ladder closed with parent promotion |
| `artifact_integrity_status: verified-at-completion` | Structural export rehash accepted |

## Claim language

After a green ladder you may claim: **this exact Path Beta tuple reached
`measured-run-pass` on this host class with structural PEFT export verification.**

You may **not** claim: semantic adapter reload, quality, multi-GPU, other cards,
or release readiness without separate evidence.

## Evidence

Current-HEAD acceptance packet:

[`docs/operations/evidence/2026-08-12-path-beta-cuda-lora-m4/`](../operations/evidence/2026-08-12-path-beta-cuda-lora-m4/)

Historical identity freeze (not current-HEAD):

[`docs/operations/evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/`](../operations/evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/)
