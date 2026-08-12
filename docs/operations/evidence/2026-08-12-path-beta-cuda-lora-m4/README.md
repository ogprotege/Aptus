# Path Beta CUDA LoRA current-HEAD acceptance, 2026-08-12

> **Status:** Passed — one clean managed five-job ladder to `measured-run-pass`  
> **Evidence class:** Exact-scope Path Beta runtime acceptance at current source  
> **Path ID:** `path-beta-cuda-lora-single-v1`  
> **Base source:** `93d69f63c7d3c1147ce186e810c355cdcf1a1b9c` (main tip at measurement)  
> **Product fix included:** CUDA PEP 440 local-label pin match in `cuda/preflight.py`  
> **Last reviewed:** 2026-08-12  
> **Review by:** Before any Path Beta claim broadening, CUDA pin change, or host-class change

## Result

One managed workflow completed dependency → model-data → measured preflight →
two-phase pilot (checkpoint continuation observed) → confirmed full train with
parent promotion to **`measured-run-pass`** for the frozen Path Beta identity on
the campaign host class (Ubuntu 24.04.4 + NVIDIA RTX 3050).

| Gate | Job | Result |
| --- | --- | --- |
| dependency | `job_354b87c1746a4988b67d126a88a52c98` | completed, return code 0 |
| model-data | `job_7e135284319d422f8d94543e6d6d09cc` | 134,515,008 parameters, 210 adapter targets, 4 rows |
| preflight | `job_b89b263bbc4b458bb8865a40ccb2f540` | synthetic peak 17,260,032 bytes |
| pilot | `job_f232b5a638a94a4a87dc711d098fff87` | steps 1 then 2; checkpoint continuation observed |
| train | `job_54ee79cb3ddc43c3908d691e73527beb` | global step 3; structural PEFT export; `verified-at-completion` |

Terminal validation report state: **`measured-run-pass`**  
(`measured_run_completed_at` `2026-08-12T14:14:44.967035+00:00`).

## Bound inputs

| Field | Value |
| --- | --- |
| Model | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| Revision | `12fd25f77366fa6b3b4b768ec3050bf629380bac` |
| Dataset | `examples/support-sft.jsonl` |
| Dataset SHA-256 | `bf2dca3d6398d639f47a883203920e1f52b0981becac96734147054e53f8aa44` |
| Plan schema | `aptus.training-plan.v6` |
| Bundle schema | `aptus.bundle.v3` |
| Plan ID | `plan_6870eaf879c843dd0ede` |
| Candidate ID | `cand_2fe2c0a05360293358f6` |
| Method / placement | BF16 LoRA / `single` |
| Artifact fingerprint | `1a41e586511cff2cf68b1e0794a9b1b57395601a072fc4661bf0ebff140bf855` |
| Policy snapshot | `c2ae989c8b68df6e984dc7c8670397e791ff30e1f5ce82129e25c1c2b93268d8` |
| Runtime | Python 3.12.3, Torch 2.13.0+cu130, Transformers 5.14.1, Accelerate 1.14.0, Safetensors 0.8.0, PEFT 0.19.1 |
| Host class | Ubuntu 24.04.4 LTS, NVIDIA GeForce RTX 3050, 8,220,573,696 bytes Torch-visible VRAM, driver 595.84, compute 8.6 |

## Export verification

Structural PEFT adapter export (23,123,132 bytes total export tree).  
`adapter_model.safetensors` SHA-256:

`5aab8b259824a1dc81613c01e6ea49cb2d757e9601d5f080c009aabef9eafffa`

Parent recorded `artifact_integrity_status: verified-at-completion`.  
**Semantic CUDA adapter reload is not claimed.**

## Product fix required for the ladder

The first dependency attempt failed because official CUDA wheels report
`torch==2.13.0+cu130` while the bundle pin is `torch==2.13.0`. Aptus now compares
**PEP 440 public versions** in the CUDA portable dependency gate so local labels
are accepted when the release pin matches. Wrong public versions still fail closed.

## M4.4 Job-control cancel smoke

Owned process-group cancellation on the same host/class (pilot job) recorded in
[`m4.4-cancel-smoke.json`](m4.4-cancel-smoke.json):

| Field | Value |
| --- | --- |
| Job | `job_70d112ebc3fc4e1f808f91ec0e0cd548` |
| Final state | **`cancelled`** (not `completed`) |
| Return code | `-15` (SIGTERM) |
| Cancel reason | `M4_4_CANCEL_SMOKE` |
| Milestones | cancel_requested → process_group_terminated → lease_reconciled |
| Parent success claim | none (`artifact_integrity_status` null; no completion attestation) |

Cancellation does **not** report success. Host-global lease was reconciled after
process-group termination.

## Claim boundary

**Supports only** the exact tuple above (source + pin fix, host class measured
here, clean-env runtime pins, model revision, dataset digest, plan/candidate,
fingerprint).

**Does not support:** semantic CUDA adapter reload; DDP/FSDP/multi-GPU; model
quality/safety; throughput; other CUDA cards; public release readiness;
historical fingerprint transfer from the August 6 freeze packet.

The historical packet
[`2026-08-06-smollm2-cuda-lora-single-acceptance`](../2026-08-06-smollm2-cuda-lora-single-acceptance/)
remains the identity freeze and historical baseline. Its bundle fingerprint
`296fb7b7…` is **not** transferred to this compile.

## Operator procedure

Follow [Path Beta operator runbook](../../../guides/path-beta-cuda-lora-operator.md).

## Files

- [`acceptance-summary.json`](acceptance-summary.json) — machine-readable rollup  
- [`SHA256SUMS`](SHA256SUMS) — digests of committed packet files  

Raw job state, HF caches, adapter binaries, and absolute host paths remain
outside Git (bound by digests in the summary where applicable).
