# Qwen2 Phase 6 MLX-LM acceptance, 2026-08-05

> **Status:** Passed — two clean `measured-run-pass` repetitions
>
> **Acceptance source:** `14ed44b52a76bb84d8d9db4f2303951aa641339b`
>
> **Static policy source:** `81bb1a286a45a5d5b424288699f8acdd8c051ecf`
>
> **Scope:** Exact pinned artifact, source, host, runtime, dataset, and policy snapshot
>
> **Not a claim:** Model quality, general Qwen2 compatibility, CUDA acceptance, or production throughput
>
> **Last reviewed:** 2026-08-05
>
> **Review by:** On any MLX runtime, compiler, admission, evidence, or parent-promotion change

## Result

Aptus completed the current training-plan v5 and bundle v3 MLX-LM QLoRA
ladder twice from a clean detached checkout. Each repetition independently ran
dependency validation, exact model/data validation, measured preflight, an
uninterrupted real-model pilot, confirmed full training, immutable adapter
export, fresh-process reload, and fresh-process job reconciliation.

Both repetitions ended with five completed jobs, return code `0`, verified
artifact integrity, a `measured-run-pass` completion attestation, and a final
`measured-run-pass` validation report.

| Repetition | Preflight ID / peak | Pilot ID / peak | Full job / run | Full peak |
| --- | --- | --- | --- | ---: |
| 1 | `bounded-smoke_6684192ba48b45ddb4065f0f498f4c5d` / 515,116,438 B | `pilot_1e1acf50837d4a4281dab527c618cd95` / 521,537,590 B | `job_23628ccc106047a580e802324cf8dd57` / `run_23628ccc106047a580e802324cf8dd57` | 581,760,722 B |
| 2 | `bounded-smoke_2657a4c9ff744f0c97900ccf47592bde` / 521,389,026 B | `pilot_93808504c439463083f27af6b70274a9` / 521,461,772 B | `job_cd5512825a844a89839c9312adab7f94` / `run_cd5512825a844a89839c9312adab7f94` | 582,001,282 B |

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
- Bundle fingerprint: `f1d175193792e2b09c606f92c8db1d58e0a7c4bcb531c03c76fc71ad2be10b9e`
- Bundle ZIP SHA-256: `0a7a912b97f82f2c3f7b9069281c8d49ca6543a3894ab050840549c3b258b664`

The two embedded plans, manifests, and ZIP archives were byte-identical. The
ten cached model-file hashes still matched the retained July manifest before
and after execution, the model size matched, and no incomplete cache file was
present. Runtime execution was offline after one fresh metadata inspection.

## Parent-owned completion proof

The acceptance source includes the MLX managed-completion correction discovered
by the first diagnostic attempt. A current managed runner now records an exact
`execution-approved` handoff before training and defers terminal promotion to
the Aptus parent. The parent verifies the immutable run artifacts, source report
hash, active-run binding, and completion evidence before writing
`measured-run-pass` with `aptus.parent-promotion.v1`.

Both qualifying commands contain `--defer-parent-promotion`. For each run:

- the verified child source state was exactly `execution-approved`;
- the active run bound the expected plan, candidate, run ID, and output root;
- the reconstructed pre-promotion report matched its stored source SHA-256;
- the receipt job ID, run ID, and bundle fingerprint matched the job record;
- `promoted_at` equaled `measured_run_completed_at`;
- the receipt evidence SHA-256 independently recomputed from canonical stored
  evidence; and
- no active or pending full-run fields remained after promotion.

Run 1 receipt evidence SHA-256 is
`6abecc89be8159edda9235aa957b5a99447501ed4454ece446d9f03557d7c0c1`;
run 2 is
`a3603a2e30bf3981c0f2971ccebe15d72cff548013c75fd6303460dc0ff8d42b`.
The sanitized pre-promotion reports and reconciliations retain the semantic
records; `raw-artifact-digests.json` retains the hashes of the original
temporary bytes.

## Deterministic runtime evidence

Both full runs produced identical learned adapter weights:

- final `adapters.safetensors` SHA-256:
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

Run IDs, timestamps, measured available memory, absolute temporary paths, and
adapter configuration metadata are expected to differ. The 240,560-byte full
peak difference is about 0.041%. The learned safetensors are byte-identical.

## Host and runtime

- Apple M5 Pro, arm64, 64 GiB unified memory
- macOS 26.6, build 25G72
- Python 3.12.13
- MLX 0.31.2
- MLX-LM 0.31.3
- MLX Metal 0.31.2

The full-run required-available threshold was 13,048,052,182 bytes, including
an 8,589,934,592-byte reserve. The plan point estimate was 2,372,989,999 bytes
and upper estimate was 4,457,927,670 bytes. Both runs retained substantial
unified-memory and disk headroom.

No recorded acceptance process remained at handoff and the host-global Aptus
GPU lease was absent. The detached checkout remained clean at the exact source
commit. Adapter binaries, model files, raw job state, process identifiers,
logs, environment secrets, and generated text were not committed.

## Nonqualifying diagnostic

Attempt 0 at the static policy commit reproduced a preexisting managed-workflow
defect: the child completed training and reload with return code `0`, but the
JobService outcome failed because terminal `measured-run-pass` lacked a valid
parent receipt. It is excluded from the acceptance count. See
[`diagnostics/attempt-01-unreceipted-parent-promotion/`](diagnostics/attempt-01-unreceipted-parent-promotion/README.md).

## Reproduction and retained records

The sanitized command shape, environment flags, action ordering, and retention
rules are in [`acceptance-procedure.json`](acceptance-procedure.json).
The portable compiled plan is [`clean-plan.json`](clean-plan.json), with
[`bundle-manifest.json`](bundle-manifest.json),
[`model-policy-snapshot.v1.json`](model-policy-snapshot.v1.json), and
[`split-contract.json`](split-contract.json).

Each `runs/run-N/` directory contains:

- a five-action `jobs-summary.json`;
- the reconstructed and path-normalized `pre-promotion-source-report.json`;
- a sanitized `train-reconciliation.json` with receipt rehash result;
- the terminal `validation-report.json` projection;
- full metrics, final export, final artifact manifest, and reload evidence.

`acceptance-summary.json` is the machine-readable rollup.
`raw-artifact-digests.json` identifies uncommitted source records.
`SHA256SUMS` binds every committed evidence file other than itself.

## Evidence boundary

This record closes the current-source Phase 6 v5/v3 MLX-LM runtime gate for the
exact pinned Qwen2.5 artifact on this host and runtime. The reviewed policy is
still a configuration-footprint policy, not an artifact allowlist: a different
matching Qwen2 artifact remains conditional and must complete its own required
model-data, measured-preflight, and pilot gates. This evidence does not qualify
CUDA, establish model quality, or promise production-scale performance.
