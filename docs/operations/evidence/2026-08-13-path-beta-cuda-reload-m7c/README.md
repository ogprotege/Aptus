# Path Beta CUDA adapter reload, 2026-08-13 (M7-C)

> **Status:** Passed — fresh-process PEFT reload + 1–4 token generation  
> **Evidence class:** Exact-scope Path Beta semantic CUDA adapter reload  
> **Path ID:** `path-beta-cuda-lora-single-v1`  
> **Expansion axis:** M7-C only (not a second model or second host)  
> **Compile source:** `36bef48d6ca3c0b11bf39da823ae4bc24f4c94fb`  
> **Last reviewed:** 2026-08-13  
> **Review by:** Before transferring this claim to another compile, host, or method

## Result

A current-HEAD Path Beta ladder on Sherminator reached
`measured-run-pass`, then a **separate child process** loaded the pinned base
plus the PEFT `final/` adapter and generated **4** tokens.

| Gate | Job | Result |
| --- | --- | --- |
| dependency | `job_af3f06cef1e34683aad136379e1ec210` | completed, rc 0 |
| model-data | `job_07f967b564b543bcb3810a486db55e16` | 134,515,008 parameters, 210 adapter targets |
| preflight | `job_bb03b36103e34fc4b9103e2c07e9015b` | synthetic peak 17,260,032 bytes |
| pilot | `job_4124c077a8784c109825858bf61cfbaf` | two-phase checkpoint continuation |
| train | `job_5834c4cc5e2e46ffb9e831198e4be2df` | global step 3; `measured-run-pass` |
| reload | child pid 10974 / parent 10973 | `aptus.cuda-reload-evidence.v1` |

## Bound inputs

| Field | Value |
| --- | --- |
| Model | `HuggingFaceTB/SmolLM2-135M-Instruct` @ `12fd25f77366fa6b3b4b768ec3050bf629380bac` |
| Dataset SHA-256 | `bf2dca3d6398d639f47a883203920e1f52b0981becac96734147054e53f8aa44` |
| Plan / candidate | `plan_5eb5d3e3a0675a62a326` / `cand_11ef673a72fe27410ea3` |
| Artifact fingerprint | `cf7858e5d466df4b1a9c723a84db1ed5acd9907294555978e73ddeef99786fb4` |
| Run | `run_5834c4cc5e2e46ffb9e831198e4be2df` |
| Adapter SHA-256 | `5aab8b259824a1dc81613c01e6ea49cb2d757e9601d5f080c009aabef9eafffa` |
| Reload peak | 297,857,024 bytes |
| Generation tokens | 4 |
| Generation text SHA-256 | `33857045b46c5a893c8f634bcde5a158fbb8a12feebfcb91589ffdbb09faaeb3` |
| Host | Sherminator, Ubuntu 24.04.4, RTX 3050, 8,220,573,696 Torch-visible bytes, driver 595.84 |
| Runtime | Python 3.12.3, Torch 2.13.0+cu130, Transformers 5.14.1, PEFT 0.19.1 |
| Verifier | `src/aptus/_bundle_programs/cuda/reload.py` |

CUDA `measured-run-pass` still does **not** require this reload for promotion.
The verifier is a measured M7-C program, not yet a CUDA parent gate.

A prior reload of the M4 adapter (same safetensors digest) is supporting only
and is not this compile’s identity.

## Claim boundary

**Supports only:** this Path Beta tuple, this compile fingerprint, this host
class, this fresh-process PEFT load, and 1–4 generated tokens. Inference from
the saved adapter. Not training resume.

**Does not support:** M7-A / M7-B; other models, hosts, or methods; quality;
throughput; making all CUDA cells claim reload; changing `measured-run-pass`.

## Files

- [`reload-evidence.json`](reload-evidence.json)  
- [`SHA256SUMS`](SHA256SUMS)  
