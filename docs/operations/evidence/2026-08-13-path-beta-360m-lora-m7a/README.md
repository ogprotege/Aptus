# Path Beta runtime, second model (SmolLM2-360M LoRA), 2026-08-13 (M7-A)

> **Status:** Passed — one clean managed ladder to `measured-run-pass`  
> **Evidence class:** Exact-scope second model on Path Beta CUDA runtime  
> **Expansion axis:** M7-A only  
> **Compile source:** `36bef48d6ca3c0b11bf39da823ae4bc24f4c94fb`  
> **Last reviewed:** 2026-08-13  
> **Review by:** Before transferring this claim to another model, host, or method

## Result

| Gate | Job | Result |
| --- | --- | --- |
| dependency | `job_cb37a1b78ca549c8999b940138a732ee` | completed, rc 0 |
| model-data | `job_af848debfe9a4498b39bbf11e5a096fa` | 361,821,120 parameters, 224 adapter targets |
| preflight | `job_c021017e3c5d4404bf1dad7681662382` | synthetic peak 17,260,032 bytes |
| pilot | `job_97b944999e5c41a8802ea59e609a5375` | two-phase continuation |
| train | `job_2b1ce8808ff34e6e94528843899405eb` | global step 3; `measured-run-pass` |

## Bound inputs

| Field | Value |
| --- | --- |
| Model | `HuggingFaceTB/SmolLM2-360M-Instruct` @ `a10cc1512eabd3dde888204e902eca88bddb4951` |
| Dataset SHA-256 | `bf2dca3d6398d639f47a883203920e1f52b0981becac96734147054e53f8aa44` |
| Plan / candidate | `plan_73a45f2660d8b869922a` / `cand_083ecc2b84e62619090b` |
| Fingerprint | `e783adb7e45d41f7470235ee570b9e0a22101ad151c7ce0886cba5f8462d6680` |
| Adapter SHA-256 | `fc88e5a77d7304f44712c24d1fce2a147cd1810fc4ee84d7f668ebf01e1377a3` |
| Host | Sherminator, Ubuntu 24.04.4, RTX 3050 (same class as Path Beta 135M) |
| Runtime | Python 3.12.3, Torch 2.13.0+cu130, Transformers 5.14.1, PEFT 0.19.1 |

## Claim boundary

**Supports only** this 360M LoRA single tuple on this host class.

**Does not support:** Path Beta 135M transfer; M7-B; 360M Full; quality; reload unless separately recorded; other hosts.
