# M7-C identity freeze — semantic CUDA adapter reload

> **Status:** Frozen for M7-C  
> **Path ID:** `path-beta-cuda-lora-single-v1` (unchanged)  
> **Expansion:** fresh-process PEFT adapter reload + 1–4 token generation  
> **Frozen:** 2026-08-13  
> **Does not expand:** model artifact, host class, method

## Identity (same as Path Beta)

| Field | Value |
| --- | --- |
| Model | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| Revision | `12fd25f77366fa6b3b4b768ec3050bf629380bac` |
| Dataset | `examples/support-sft.jsonl` |
| Dataset SHA-256 | `bf2dca3d6398d639f47a883203920e1f52b0981becac96734147054e53f8aa44` |
| Method / placement | LoRA BF16 / `single` |
| Runtime | `transformers-peft-cuda` |
| Host class | Ubuntu 24.04.4 LTS + NVIDIA GeForce RTX 3050 (~8 GiB) |
| Authorized host | Sherminator, `wts@192.168.1.12` |

## Reload proof (new)

| Field | Value |
| --- | --- |
| Schema | `aptus.cuda-reload-evidence.v1` |
| Process | Fresh child: `verifier_pid != parent_pid`, both > 0 |
| Load | Pinned base revision + PEFT adapter from owned `final/` |
| Generate | Prompt `Aptus adapter reload verification:`, **1–4** tokens |
| Peak | `torch.cuda.max_memory_allocated()` > 0 |
| Bindings | `plan_id`, `candidate_id`, model revision, dataset SHA-256, adapter file SHA-256s from `final-export.json` |

## Explicit non-claims

- Not a second model (M7-A) or second host (M7-B)
- Not model quality, safety, or throughput
- Not training resume / optimizer restore
- Not DDP, FSDP, Full, QLoRA, or int8-LoRA
- Not a change to CUDA `measured-run-pass` promotion until a later contract bump
- Historical M4 adapter reload, if recorded, is not transferred to a later compile without a fresh run
