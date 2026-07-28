# Qwen3 30B-A3B MoE admission and performance evidence

> **Status:** Blocked at live unified-memory admission | **Authority:** Immutable target-host attempt record | **Applies to:** Exact Qwen3 MoE compatibility row | **Audience:** Release reviewers and operators | **Last reviewed:** 2026-07-28 | **Review by:** Any MoE runtime, memory formula, or acceptance-contract change

> **Host:** Apple M5 Pro, 64 GiB | **Recorded:** 2026-07-28 UTC | **Source:** `35f80499918a8e7f06b8076dfd95b2985b1a765a`

## Decision

The exact Qwen3 MoE contract passed plan generation, deterministic compilation,
static validation, dependency validation, offline revision resolution,
configuration binding, and packed-checkpoint measurement. Model-data validation
then refused execution before `mlx_lm.utils.load` could load any weights.

The host had 30,952,833,024 bytes of available unified memory. Aptus required
51,280,521,945 bytes after measuring the packed checkpoint and retaining the
8 GiB reserve. The exact shortfall was 20,327,688,921 bytes, or 18.932 GiB.

This is a safe admission block. It is not a passing 30B model-data gate, pilot,
training run, reload, export, throughput measurement, or quality result.

Machine-readable records:

- [admission-summary.json](admission-summary.json)
- [performance-summary.json](performance-summary.json)
- [SHA256SUMS](SHA256SUMS)

## Bound configuration

| Item | Exact value |
| --- | --- |
| Model | `mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit` |
| Revision | `e9675aa3ca5f900ccef55267914466d55ab325fa` |
| Provider contract | `qwen3_moe`, `Qwen3MoeForCausalLM` |
| Total resident parameters | 30,532,122,624 |
| Derived active parameters per token | 3,353,032,704, or 10.982% |
| Expert topology | 128 routed experts, 8 selected per token, width 768, 48 sparse layers |
| Shared expert | Absent |
| Quantization layout | Four-bit group-64 default and one eight-bit group-64 router-gate override for each of 48 layers |
| Method | Single-device MLX-LM QLoRA |
| Adapter scope | `q_proj`, `k_proj`, `v_proj`, and `o_proj` |
| Sequence and batch | 64 tokens, effective batch 1 |
| Dataset digest | `bf2dca3d6398d639f47a883203920e1f52b0981becac96734147054e53f8aa44` |
| Runtime | Python 3.12.13, MLX 0.31.2, MLX-LM 0.31.3 |
| Host | Apple M5 Pro, 64 GiB, macOS 26.6 build 25G72 |

The derived active count is a routed-compute fact. It never replaces total
parameters in weight residency, checkpoint size, staging, or disk estimates.

## Gate ledger

| Gate | Result | Evidence |
| --- | --- | --- |
| Plan | Pass | `plan_26d00f729ee3ff7ec867` |
| Compile and static validation | Pass | Bundle fingerprint `39a734df1d20852b811f427006a13c56503b85c87fb7a28d4d9463e83b2bfde2` |
| Dependency | Pass | Job `job_c8a71ea48fa6439bb20e5bf45f66ae4b` |
| Model-data | Blocked before model load | Job `job_e9a814823b7b46008ee2c0cefb1467cb` |
| Measured preflight | Not run | Blocked by model-data admission |
| Pilot | Not run | Blocked by model-data admission |
| Fresh adapter reload | Not run | No adapter exists |
| Confirmed training and export | Not run | No pilot authorization exists |

The generated validator validates the pinned configuration before calling the
admission function. The admission function measures checkpoint shards and live
memory before the model loader call. The recorded stack trace stopped inside
that admission function.

## Memory result

| Measurement | Bytes | GiB |
| --- | ---: | ---: |
| Planned packed weight residency | 17,180,610,432 | 16.001 |
| Observed four-shard safetensors size | 17,181,071,994 | 16.001 |
| Packed-size adjustment | 461,562 | 0.000 |
| Adjusted point estimate | 28,481,599,618 | 26.526 |
| Adjusted upper estimate | 42,690,587,353 | 39.759 |
| Required Aptus reserve | 8,589,934,592 | 8.000 |
| Required available memory | 51,280,521,945 | 47.759 |
| Available at the gate | 30,952,833,024 | 28.827 |
| Shortfall | 20,327,688,921 | 18.932 |

The observed checkpoint exceeded the planned packed residency by only 461,562
bytes. That is within the one-part-in-ten-thousand packed-checkpoint tolerance.
The refusal came from live unified-memory capacity, not checkpoint identity or
layout drift.

No process or user application was terminated to manufacture headroom. A safe
retry requires a normal reboot or user-directed application shutdown, then a
fresh model-data action. The same bundle must still remeasure the checkpoint and
live memory. A passing retry would authorize only measured preflight, not pilot
or full training.

## Measured timing

The final exact attempt recorded:

| Stage | Wall time | Managed job time |
| --- | ---: | ---: |
| Plan generation | 0.11 s | Not managed |
| Compile plus static validation | 0.15 s | Not managed |
| Dependency gate | 1.18 s | 92.421 ms |
| Model-data through safe refusal | 1.70 s | 769.325 ms |

A separate fresh-process benchmark on the same host measured these warm
medians:

| Control-plane operation | Warm median |
| --- | ---: |
| Plan generation | 53.117 ms |
| Compile plus static validation | 102.745 ms |
| Standalone static validation | 83.003 ms |
| Direct generated dependency gate | 68.047 ms |
| Managed dependency job | 97.534 ms |
| Packed-checkpoint scan plus live admission | 1.950 ms |
| Offline cache resolution, config binding, and admission | 177.213 ms |

These are orchestration and validation timings. They are not model training or
generation speeds.

## Real MLX synthetic MoE probe

MLX-LM 0.31.3's real `qwen3_moe.Model` ran a small, unquantized synthetic
forward probe with two sparse layers, four experts, and two experts selected per
token. Thirty warm runs produced:

- 0.877 ms median forward time;
- 0.808 to 1.178 ms range;
- input shape `[1, 8]` and logits shape `[1, 8, 151936]`;
- 161,754,000 bytes peak MLX memory; and
- 0.112 ms median logical parameter census.

The derived 9,118 parallel positions per second is not autoregressive
generation speed. The probe is not quantized and does not project 30B training
or inference performance. It proves only that the real MLX Qwen3 MoE layer and
census path execute on this runtime.

No defensible 30B tokens per second, optimizer-step latency, peak MLX memory,
export time, reload time, or model-quality metric exists because the live gate
correctly stopped before model loading.

## Artifact identity

| Artifact | SHA-256 |
| --- | --- |
| Plan JSON | `948b15c63635ac25fb4979c952aeac7a198cdd6e5e5bb2604361c9ad731b44c1` |
| Bundle manifest | `39a734df1d20852b811f427006a13c56503b85c87fb7a28d4d9463e83b2bfde2` |
| Bundle ZIP | `cbc750cd45bae49214318c2a1008a24afe444e3a9e297b59b3d43ef849d293a8` |
| Dataset | `bf2dca3d6398d639f47a883203920e1f52b0981becac96734147054e53f8aa44` |

The large model cache, bundle, ZIP, and raw logs remain local review artifacts.
They are not committed to Git. This directory retains the compact identity,
gate result, timing, and claim boundary needed for review.

## Reproduction boundary

Offline reruns must set:

```bash
export HF_HUB_CACHE=/private/tmp/aptus-moe-acceptance-20260727/hf-cache
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export APTUS_MLX_PYTHON=/private/tmp/aptus-mlx-clean-acceptance-20260727.9gos44/runtime-env/bin/python
```

The retained directory is already a Hugging Face hub-cache root. Setting
`HF_HOME` to that directory is incorrect because the client would search for
an absent nested `hub` directory.

The exact run used `aptus spec-plan`, `aptus compile`, then managed
`dependency` and `model-data` actions. It did not invoke measured preflight,
pilot, train, or reload after admission refused execution.
