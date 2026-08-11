# M0 — Program non-goals freeze

> **Status:** Frozen — owner accepted 2026-08-11  
> **Phase:** M0  
> **Date:** 2026-08-11  

These non-goals bind the mission integrity program until a later `DECISION-*`
explicitly opens one of them. They implement KISS and depth-before-breadth.

## Frozen non-goals (program duration)

| ID | Non-goal | Rationale |
| --- | --- | --- |
| NG-01 | No DoRA / BitFit / AdaLoRA / ShareLoRA / LoReFT / AFLoRA / BiLoRA compilers | Experimental/research remain visible only; not mission-critical for trust-when-no |
| NG-02 | No ROCm / CPU training paths | Outside dual-runtime Alpha/Beta promise |
| NG-03 | No cloud runner / cost marketplace product | Solo operator + own/rented host only |
| NG-04 | No MCP / external automation training authorization | Local control plane first |
| NG-05 | No full-training resume / MLX crash continuation | Contract incomplete; fail closed stands |
| NG-06 | No “general MoE support” | Exact-row only; 30B training acceptance not in M0–M5 critical path |
| NG-07 | No multi-GPU campaign (DDP / LoRA FSDP proof) by default | Requires separate Section 12 decision; not Path Beta |
| NG-08 | No quality guarantee language | Eval is optional M8; never claim quality from loss alone |
| NG-09 | No public “supports all CUDA / all Apple Silicon” claims | Exact-identity evidence only |
| NG-10 | No silent dependency installation | Doctor remains probe-only |

## Allowed later only via DECISION

- M6 public notarization (distribution, not training breadth)
- M7 one-axis expansion (second model **or** second host **or** semantic reload)
- M8 evaluation contract
- Multi-GPU campaign (new program or explicit decision superseding NG-07)

## Owner sign-off

- [x] Owner accepts non-goals as written: owner chat authorization 2026-08-11
