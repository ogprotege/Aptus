# Path Alpha MLX QLoRA current-HEAD acceptance, 2026-08-12

> **Status:** Passed — two fresh clean `measured-run-pass` ladders  
> **Evidence class:** Exact-scope Path Alpha runtime acceptance at current source  
> **Path ID:** `path-alpha-mlx-qlora-v1`  
> **Acceptance source:** `f4775c01e6b8f932e11c2d665e90859d6aedbe04`  
> **Source tree:** `eba49709fe58ed72329813909048c00f1330d875`  
> **Last reviewed:** 2026-08-12  
> **Review by:** Before any Path Alpha claim broadening or MLX pin change

## Result

Two independent managed workflows completed dependency → model-data → measured
preflight → pilot → confirmed full train with parent promotion to
**`measured-run-pass`** for the frozen Path Alpha identity on this host.

| Run | State | Optimizer updates | Peak bytes | Adapter SHA-256 (prefix) |
| --- | --- | ---: | ---: | --- |
| 1 | measured-run-pass | 3 | 582,062,200 | `4717543bb38f0845…` |
| 2 | measured-run-pass | 3 | 582,029,576 | `4717543bb38f0845…` |

Learned adapter weights were **byte-identical across both runs** and match the
historical August 2025 acceptance adapter digest
`4717543bb38f084573a6f1ea2fa0638d71c1a1a38b1b2103545951e052d5f31b`.

## Bound inputs

| Field | Value |
| --- | --- |
| Model | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` |
| Revision | `53a32aee5e9447773fd2b85988395066aef3700a` |
| Dataset | `examples/support-sft.jsonl` |
| Dataset SHA-256 | `bf2dca3d6398d639f47a883203920e1f52b0981becac96734147054e53f8aa44` |
| Plan schema | `aptus.training-plan.v6` |
| Bundle schema | `aptus.bundle.v3` |
| Plan ID | `plan_b3630466fb9a25f6b08c` |
| Candidate ID | `cand_bec6f029a7417259d49c` |
| Artifact fingerprint | `ace50ce8b4defc2a3a871e4031a358e0942fb114980e487acac07c66f766ce14` |
| Policy snapshot | `c2ae989c8b68df6e984dc7c8670397e791ff30e1f5ce82129e25c1c2b93268d8` |
| Runtime | Python 3.12.13, mlx 0.31.2, mlx-lm 0.31.3, mlx-metal 0.31.2 |
| Host | Apple M5 Pro, arm64, 64 GiB unified, macOS 26.6.1 (25G76) |

## Claim boundary

**Supports only** the exact tuple above (source, host class measured here,
runtime pins, model revision, dataset digest, plan/candidate, fingerprint).

**Does not support:** other Qwen2 artifacts that merely match the configuration
footprint; CUDA; multi-GPU; model quality/safety; throughput; public release
readiness; MLX resume.

The historical packet
[`2026-08-05-qwen2-mlx-lm-exact-source-refresh`](../2026-08-05-qwen2-mlx-lm-exact-source-refresh/)
remains the identity freeze and historical baseline. Its bundle fingerprint is
**not** transferred to this compile; this packet freshly qualifies fingerprint
`ace50ce8…`.

## Operator procedure

Follow [Path Alpha operator runbook](../../../guides/path-alpha-mlx-operator.md).

## Files

- [`acceptance-summary.json`](acceptance-summary.json) — machine-readable rollup  
- [`SHA256SUMS`](SHA256SUMS) — digests of committed packet files  

Raw job state, HF caches, adapter binaries, and absolute host paths remain
outside Git (bound by digests in the summary where applicable).
