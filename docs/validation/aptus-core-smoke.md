# Aptus Core Offline Smoke Evidence

Date: 2026-07-21

## Environment

- Python 3.14.6
- PyTorch 2.13.0
- Transformers 5.14.1
- PEFT 0.19.1
- Accelerate 1.14.0
- Network-disabled mode:
  - `HF_HUB_OFFLINE=1`
  - `TRANSFORMERS_OFFLINE=1`

## Workflow exercised

1. The Aptus CLI profiled `tests/fixtures/text.jsonl`.
2. It evaluated LoRA and QLoRA against a manually supplied 7B Llama-family
   model and one 24 GiB CUDA device.
3. The quality objective selected LoRA.
4. Aptus generated `plan.json`, the shared `plan_contract.py`, `train.py`,
   `validate.py`, `requirements.txt`, `README.md`, and
   `validation-report.json`.
5. Static validation passed.
6. `train.py --validate-only` passed without ML dependencies.
7. A clean virtual environment installed the exact generated requirements.
8. `train.py --smoke` constructed a tiny local Llama configuration, attached
   PEFT LoRA adapters, completed one forward/backward/AdamW step, and reported
   finite loss `4.869252`.
9. `validation-report.json` persisted state `smoke-pass` with both
   dependency-free and optimizer-step runtime evidence, bound to a SHA-256
   fingerprint of all generated artifacts. A later `--validate-only` run did
   not downgrade the smoke state.

Observed output:

```text
Aptus offline smoke passed; loss=4.869252
```

## What this proves

- Generated Python is executable under the pinned environment.
- The current Transformers and PEFT APIs accept the generated LoRA adapter
  contract.
- The smoke path performs a real optimizer step without downloading model
  weights or data.

## What this does not prove

- The 7B model itself fits the predicted VRAM.
- QLoRA/bitsandbytes works on a CUDA GPU.
- The heuristic-v1 memory estimate is calibrated.
- Normal training, checkpointing, model download, dataset tokenization, or
  adapter quality has been validated.
- Any cloud, distributed, API, MCP, or UI workflow exists.

Those remain explicit later gates.
