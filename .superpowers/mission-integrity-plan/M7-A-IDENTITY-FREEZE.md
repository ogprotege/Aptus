# M7-A identity freeze — second model on Path Beta runtime

> **Status:** Frozen for M7-A  
> **Expands:** model artifact only  
> **Runtime:** `transformers-peft-cuda` (same as Path Beta)  
> **Host class:** unchanged (Ubuntu 24.04.4 + RTX 3050 / Sherminator)  
> **Frozen:** 2026-08-13

## Why this artifact

Campaign Phase 7 already admitted SmolLM2-360M LoRA on this host class.
That is a second **model**, not a second host or method.

## Identity

| Field | Value |
| --- | --- |
| Model | `HuggingFaceTB/SmolLM2-360M-Instruct` |
| Revision | `a10cc1512eabd3dde888204e902eca88bddb4951` |
| Family / architecture | `llama` / `LlamaForCausalLM` |
| Parameters | 361,821,120 |
| Hidden / intermediate / layers | 960 / 2560 / 32 |
| Context | 8192 |
| Dataset | `examples/support-sft.jsonl` SHA-256 `bf2dca3d…` |
| Method / placement | LoRA BF16 / `single` |
| Host | Sherminator, same class as Path Beta |

## Explicit non-claims

- Not Path Beta (135M) transfer
- Not M7-B (same host)
- Not M7-C (reload optional after train; not required for this packet unless separately recorded)
- Not 360M Full (planner rejected on this VRAM class in Phase 7)
- Not quality / multi-GPU
