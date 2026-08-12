# Path Alpha operator runbook — MLX QLoRA on Apple Silicon

> **Status:** Active | **Authority:** Path-scoped operator procedure | **Applies to:** Aptus 0.2 Path Alpha (`path-alpha-mlx-qlora-v1`) | **Audience:** Solo operators on Apple Silicon | **Last reviewed:** 2026-08-12 | **Review by:** When Path Alpha identity or MLX pins change

This runbook completes **Journey A** for the frozen Path Alpha identity: one
exact Qwen2.5 0.5B 4-bit MLX QLoRA path from facts through `measured-run-pass`.

It is **not** general Apple Silicon certification.

## Hardware prerequisites

- Apple Silicon Mac (historical acceptance host class: M5 Pro, 64 GiB unified)
- Enough free unified memory for model load + pilot peak + 8 GiB reserve
  (planning uses live `host_ram_free` and an 8 GiB Apple reserve floor)
- Tens of GiB free disk for HF cache, bundle, and run outputs

## Install Aptus (source checkout)

```bash
cd /path/to/Aptus
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip   # or: uv pip install --python .venv/bin/python -e '.[server,test]'
# If the venv has no pip, use uv as in CONTRIBUTING / local tooling.
python -m pip install -e '.[server,test]'
# Or with uv:
# uv pip install --python .venv/bin/python -e '.[server,test]'
uv pip install --python .venv/bin/python 'mlx==0.31.2' 'mlx-lm==0.31.3'
```

Pinned runtime: **mlx==0.31.2**, **mlx-lm==0.31.3** (and mlx-metal matching).

```bash
export PYTHONPATH=src:.
export APTUS_MLX_PYTHON="$(pwd)/.venv/bin/python"
python -m aptus doctor
```

Doctor must report at least one **compatible** `mlx-lm` interpreter. Aptus does
not install packages for you.

## Exact Path Alpha identity

| Field | Value |
| --- | --- |
| Path ID | `path-alpha-mlx-qlora-v1` |
| Model | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` |
| Revision | `53a32aee5e9447773fd2b85988395066aef3700a` |
| Family / type / arch | `qwen` / `qwen2` / `Qwen2ForCausalLM` |
| Layers / hidden / intermediate | 24 / 896 / 4864 |
| Parameters (declared B) | `0.494` |
| Context | 32768 |
| Quantization | 4-bit, group size 64, uniform |
| Dataset | `examples/support-sft.jsonl` |
| Dataset SHA-256 | `bf2dca3d6398d639f47a883203920e1f52b0981becac96734147054e53f8aa44` |
| Method / placement | QLoRA / `single` |
| Runtime | `mlx-lm` on `mps` |
| Expected plan status | **conditional** (pilot-required) |

Verify dataset digest:

```bash
shasum -a 256 examples/support-sft.jsonl
```

## Plan

Measure host headroom (or pass measured values from `aptus hardware`):

```bash
python -m aptus hardware
```

```bash
WORKDIR=./aptus-work/path-alpha
mkdir -p "$WORKDIR"
FREE_GIB=22   # replace with measured host_ram_free_bytes / 1024**3
VRAM_GIB=51.84  # Metal working-set ceiling from hardware probe

python -m aptus spec-plan \
  --model-id mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --revision 53a32aee5e9447773fd2b85988395066aef3700a \
  --family qwen --parameters-b 0.494 \
  --model-type qwen2 --architecture Qwen2ForCausalLM \
  --quantization-bits 4 --quantization-group-size 64 \
  --hidden-size 896 --intermediate-size 4864 --layers 24 \
  --context-length 32768 --license apache-2.0 --confirm-training-allowed \
  --dataset ./examples/support-sft.jsonl --sample-limit 64 \
  --backend mps --training-runtime mlx-lm --gpu-count 1 \
  --vram-gib "$VRAM_GIB" --host-ram-gib 64 --host-ram-free-gib "$FREE_GIB" \
  --reserve-gib 8 --disk-free-gib 400 \
  --objective memory --sequence-length 128 --effective-batch-size 1 \
  --epochs 1 --prefer-method qlora --optimizer-steps 3 \
  --output "$WORKDIR/plan.json"
```

Expect **recommended** `qlora` / `single` / **conditional**. Stderr prints refusal
guidance for non-viable rows (M2). Plan schema is `aptus.training-plan.v6`.

## Compile

```bash
BUNDLE="$WORKDIR/bundle-alpha"
rm -rf "$BUNDLE"   # must be empty or nonexistent
python -m aptus compile --plan "$WORKDIR/plan.json" --output "$BUNDLE"
```

No-clobber: never recompile into a non-empty directory. Note `plan_id`,
`candidate_id`, and `policy_snapshot_sha256` from `bundle-manifest.json`.

## Ordered runtime actions

Use a private state directory and the same external MLX interpreter:

```bash
STATE=./aptus-work/path-alpha-state
export APTUS_MLX_PYTHON="$(pwd)/.venv/bin/python"
export PYTHONPATH=src:.

python -m aptus run "$BUNDLE" --action dependency --state-dir "$STATE"
python -m aptus run "$BUNDLE" --action model-data --state-dir "$STATE"
python -m aptus run "$BUNDLE" --action preflight --state-dir "$STATE"
python -m aptus run "$BUNDLE" --action pilot --state-dir "$STATE"
python -m aptus run "$BUNDLE" --action train --confirm-full-train --state-dir "$STATE"
```

| Action | Success looks like |
| --- | --- |
| dependency | Pins import: mlx 0.31.2, mlx-lm 0.31.3 |
| model-data | Model+tokenizer load; all train/valid rows tokenized; 4-bit metadata verified |
| preflight | Bounded optimizer work; positive MLX peak and adapter delta |
| pilot | ≥2 optimizer updates; finite losses; fresh-process reload 1–4 tokens; `pilot-pass` |
| train | Confirmed full duration; parent promotion; **`measured-run-pass`** |

First model-data may download ~290 MB weights into the HF cache. Prefer offline
reruns after the cache is warm (`HF_HUB_OFFLINE=1`) when re-proving.

## Failure appendix (catalog codes)

| Code | Operator action |
| --- | --- |
| `conditional_pilot_required` | Expected on plan; not a hard refuse — complete pilot |
| `infeasible_memory` / live admission refuse | Free unified memory; lower batch/seq; do not disable reserve |
| `full_fp16` / `mlx_full` | Expected for non-QLoRA rows under this policy |
| `replan_required` | Policy/registry changed; replan and new bundle path |
| Doctor incompatible | Fix interpreter pins; never let Aptus install for you |

## What this runbook does **not** claim

- Not every Qwen2 / 24-layer / 4-bit artifact  
- Not CUDA, multi-GPU, quality, or throughput  
- Not public release readiness or notarization  
- Not transfer of a historical bundle fingerprint to a new compile  

## Related

- [Mission integrity plan](../product/mission-integrity-plan.md)  
- [Mission integrity plan](../product/mission-integrity-plan.md) (Path Alpha freeze tables)  
- [Compile / validate / run](compile-validate-run.md)  
- [Troubleshooting](troubleshooting.md)  
- [2026-08-12 Path Alpha evidence](../operations/evidence/2026-08-12-path-alpha-mlx-m3/README.md)  
