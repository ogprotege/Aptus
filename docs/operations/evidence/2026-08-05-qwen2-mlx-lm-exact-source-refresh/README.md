# Qwen2 MLX-LM exact-source refresh, 2026-08-05

> **Status:** Passed — two fresh, clean `measured-run-pass` repetitions
>
> **Documentation status:** Active exact-source evidence supplement
>
> **Acceptance source:** `719255153e3fc7e38e83b5ff826d587e5e58bf80`
>
> **Source tree:** `be99f5664ccb580f2600471f1ae3241a294b1a7e`
>
> **Scope:** Exact pinned artifact, source, host, runtime, dataset, plan, policy snapshot, and refreshed bundle fingerprint
>
> **Not a claim:** CUDA acceptance, general Qwen2 compatibility, model quality or safety, performance or throughput, production readiness, or release readiness
>
> **Last reviewed:** 2026-08-05
>
> **Review by:** Any MLX runtime, compiler, generated-operator-document, admission, evidence, or parent-promotion change

## Result

This supplement binds the manifested operator-document refresh at source
`719255153e3fc7e38e83b5ff826d587e5e58bf80`. Aptus compiled the retained Phase
6 plan twice and then completed two fresh, independent v5/v3 MLX-LM QLoRA
workflows. Each workflow ran dependency validation, exact model/data
validation, measured preflight, an uninterrupted real-model pilot, confirmed
full training, immutable adapter export, fresh-process reload, and parent-owned
job reconciliation.

All ten managed jobs completed with return code `0`. Both full runs ended with
verified artifact integrity, a valid `aptus.parent-promotion.v1` receipt, a
terminal `measured-run-pass` report, and no pending or active run fields.

| Repetition | Preflight ID / peak | Pilot ID / peak | Full job / run | Full peak |
| --- | --- | --- | --- | ---: |
| 1 | `bounded-smoke_8acddeda2e9f424b8cd47a70d6dd64bf` / 521,459,856 B | `pilot_bf7329b400474a3d8175b14f94e3526d` / 521,476,024 B | `job_24387e5ba60d43b5a2e7fdbf14f018f8` / `run_24387e5ba60d43b5a2e7fdbf14f018f8` | 581,965,116 B |
| 2 | `bounded-smoke_44480caaaef048cc895e26740bd883a5` / 521,459,856 B | `pilot_359701f2815a4eeeb226b1307baba542` / 521,481,360 B | `job_6098adf3e1c246668fdcf2eed426b894` / `run_6098adf3e1c246668fdcf2eed426b894` | 582,146,010 B |

## Why a separate record exists

The [original Phase 6 acceptance packet](../2026-08-05-qwen2-mlx-lm-acceptance/README.md)
is preserved unchanged. Its bundle fingerprint
`f1d175193792e2b09c606f92c8db1d58e0a7c4bcb531c03c76fc71ad2be10b9e`
cannot qualify a later bundle identity.

Relative to that baseline, the refreshed manifest changes exactly two paths:
generated `README.md` and `runbook.md`. The other 27 manifested paths retain
their prior size and SHA-256. In particular, `run.py`, `train.py`,
`validate.py`, `preflight.py`, `reload.py`, `runtime_lease.py`,
`plan_contract.py`, `policy_snapshot.py`, and `requirements.txt` are
byte-identical. The two fresh compilations are themselves byte-identical at:

- bundle fingerprint:
  `ca2548cf8469fb9867f1558428803b1c9f7c19f48cba754fdb602643f23d1919`;
- bundle ZIP SHA-256:
  `fcad829b4c845c6b5d1e548b293ec1107ccd7a78ea08b63bc7a1b8ca487be9b1`.

The old runtime result is not transferred to the new fingerprint. These fresh
runs independently qualify the refreshed bundle for the exact scope below.

## Bound inputs

- Model: `mlx-community/Qwen2.5-0.5B-Instruct-4bit`
- Revision: `53a32aee5e9447773fd2b85988395066aef3700a`
- Model weights: 278,064,920 bytes
- Dataset: `examples/support-sft.jsonl`
- Dataset SHA-256: `bf2dca3d6398d639f47a883203920e1f52b0981becac96734147054e53f8aa44`
- Policy: `model.qwen2-24l.mlx-qlora` version `1.0.0`
- Path: `mlx-lm.qlora.single.dense-causal-lm.v1`
- Policy snapshot SHA-256: `c2ae989c8b68df6e984dc7c8670397e791ff30e1f5ce82129e25c1c2b93268d8`
- Plan: `plan_bd33e8ab37765b7efa93`
- Candidate: `cand_ee8a9d7fc575e9eeb71b`
- Embedded plan SHA-256: `673495afd38aee3ebab44c3692fc052bdad5bb0f2c2e8f23fad11e12278dfd23`
- Host: Apple M5 Pro, arm64, 64 GiB unified memory, macOS 26.6 build 25G72
- Runtime: Python 3.12.13, MLX 0.31.2, MLX-LM 0.31.3, MLX Metal 0.31.2

The model-file manifest, 34-package runtime inventory, interpreter, dataset,
plan, inspection receipt, provider inspection, policy snapshot, and split
contract were reverified before execution. Their content hashes match the
original packet. Runtime execution remained offline.

## Deterministic runtime evidence

Both full runs produced byte-identical learned adapter weights:

- adapter SHA-256:
  `4717543bb38f084573a6f1ea2fa0638d71c1a1a38b1b2103545951e052d5f31b`;
- train losses:
  `[3.8527462482452393, 3.1543750762939453, 4.094791889190674]`;
- validation losses:
  `[4.8986897468566895, 3.818800449371338, 3.560231924057007]`;
- adapter delta L1: `5297.497747182846`;
- changed adapter tensors: `336`;
- target instances: `168` across `24` transformer layers;
- optimizer updates: `3`; and
- fresh reload generation: `4` tokens with output digest
  `b8a36d0a0bdb2d671c21e1621c6d50a184f337d826d6c2ddf28b045371db7f31`.

The parent receipt evidence hashes were independently recomputed as
`48355544d4a1f01455f4c2b96bded2d550d1a2c71217529770dcada64a6dd79e`
and
`601689102f6d8b0570a9ceb628dd56b8c45b14fb3d0183a28c3386b59cb9b59c`.
The pre-promotion source reports were also independently reconstructed and
matched their stored source hashes.

## Records and retention

- [`acceptance-summary.json`](acceptance-summary.json) is the semantic rollup.
- [`acceptance-procedure.json`](acceptance-procedure.json) records the bounded
  action order and verification rules.
- [`bundle-manifest.json`](bundle-manifest.json) is the exact refreshed manifest.
- [`bundle-comparison.json`](bundle-comparison.json) binds the baseline and
  proves the exact two-path delta.
- Each `runs/run-N/run-summary.json` is a sanitized semantic projection of one
  five-job workflow.
- [`raw-artifact-digests.json`](raw-artifact-digests.json) binds the uncommitted
  temporary source records.
- `SHA256SUMS` covers every committed file in this supplement other than itself.

The detached checkout was clean before and after both repetitions. No recorded
acceptance process or host-global Aptus GPU lease remained at handoff. Model
files, adapter binaries, ZIP archives, raw job state, logs, process identifiers,
absolute local paths, environment secrets, and generated text are not committed.

## Evidence boundary

This supplement closes the refreshed-bundle MLX-LM gate only for the exact
artifact and revision, source commit and tree, host, runtime, dataset, plan,
policy snapshot, and bundle fingerprint recorded above. The reviewed policy is
a configuration-footprint policy, not an artifact allowlist. Another matching
Qwen2 artifact must complete its own required model-data, measured-preflight,
and pilot gates. This record does not qualify CUDA, establish general Qwen2
compatibility or model quality or safety, measure production performance or
throughput, or establish production or release readiness.
