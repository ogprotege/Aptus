# SmolLM2 CUDA LoRA single-device acceptance, 2026-08-06

> **Status:** Passed — one exact CUDA LoRA single-device workflow reached `measured-run-pass`
>
> **Documentation status:** Active exact-scope runtime evidence
>
> **Acceptance source:** `c12c4d8db0037a2c278a2ad95a0a2cbda4387eed`
>
> **Source tree:** `ad482883cfb6ad2b8ac72f7b7d1009c918e5c345`
>
> **Scope:** Exact source, host, runtime, model revision, synthetic dataset, plan, policy snapshot, bundle fingerprint, and five-job sequence
>
> **Not a claim:** Repeatability, broad CUDA or model compatibility, quality or safety, benchmark performance, production readiness, or release readiness
>
> **Last reviewed:** 2026-08-06
>
> **Review by:** Any CUDA compiler, runtime, admission, pilot, export, evidence, or parent-promotion change

## Result

One exact SmolLM2 CUDA LoRA single-device acceptance completed on the recorded
Ubuntu 24.04.4 and NVIDIA RTX 3050 host. The fresh qualifying state contains
exactly five managed jobs: dependency validation, exact model/data validation,
measured synthetic preflight, a two-phase real-model pilot with confirmed
checkpoint continuation, and confirmed full training. Every job completed
with return code `0`. Each sanitized job projection retains the same accepted
runtime label, authorized policy digest, and bundle fingerprint; the bundle
manifest binds that fingerprint to the exact plan and candidate. The terminal
validation report separately binds the qualifying workflow to the accepted
environment digest.

The full run reached global step `3`, used three training rows and one
evaluation row, retained a finite LoRA-only trainable census, and produced a
23,123,131-byte structural PEFT adapter export. Parent verification rehashed
the output, recorded artifact integrity as `verified-at-completion`, promoted
the run with an `aptus.parent-promotion.v1` receipt, and left a terminal
`measured-run-pass` report with no active or pending run fields.

| Action | Managed job | Result |
| --- | --- | --- |
| Dependency | `job_ee931456c78c4137b71935c97aaea7a4` | Completed, return code 0 |
| Model/data | `job_d7539c1ec3864e31a472720dbbc097ad` | 134,515,008 parameters, 210 adapter targets, all 4 rows |
| Preflight | `job_5225b4a20dd045fb85398e94f2d478b8` | 17,260,032-byte measured CUDA peak |
| Pilot | `job_aca00b60df374907b8cec25a519866a5` | Steps 1 then 2; checkpoint continuation observed |
| Full train | `job_dad3b56e4f6d4ae1964f19839cc72b99` | Step 3, parent-verified structural adapter export |

## Bound inputs

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Immutable revision: `12fd25f77366fa6b3b4b768ec3050bf629380bac`
- Architecture and declared parameter count: `LlamaForCausalLM`, 134,515,008
- Dataset: `examples/support-sft.jsonl`, four synthetic contract rows
- Dataset SHA-256: `bf2dca3d6398d639f47a883203920e1f52b0981becac96734147054e53f8aa44`
- Plan: `plan_ed1d3fc5ecef48acf6af`
- Candidate: `cand_e0f49f5d708a27ff96ca`
- Method and placement: BF16 LoRA, `single`, world size 1, CUDA device 0
- Compiler: `transformers.peft-lora.v2`
- Runtime: `transformers-peft-cuda`
- Export: `peft-adapter-safetensors`
- Policy snapshot SHA-256: `c2ae989c8b68df6e984dc7c8670397e791ff30e1f5ce82129e25c1c2b93268d8`
- Bundle fingerprint: `296fb7b710f60345a590748f053eb15f9b5b4f4b3fec539ae3a705e31d6a640b`

## Source, runtime, and compilation

The producing checkout was detached at merge commit
`c12c4d8db0037a2c278a2ad95a0a2cbda4387eed` and tree
`ad482883cfb6ad2b8ac72f7b7d1009c918e5c345`. Its complete Python suite
passed 550 tests in 37.712 seconds. The exact source bundle and locally built
Aptus wheel are bound in `raw-artifact-digests.json`.

The accepted interpreter imported Python 3.12.3, Torch 2.13.0+cu130,
Transformers 5.14.1, Accelerate 1.14.0, Safetensors 0.8.0, and PEFT 0.19.1.
Torch observed CUDA 13.0, cuDNN 92000, BF16 support, and 8,220,573,696 bytes
of device memory. Bitsandbytes was neither a direct requirement nor installed
in the accepted runtime.

Two independent compilations produced byte-identical manifests and ZIP
archives. Their manifest/fingerprint is
`296fb7b710f60345a590748f053eb15f9b5b4f4b3fec539ae3a705e31d6a640b`;
their ZIP SHA-256 is
`6e79b01f5b723ef9b30f7bc5f886fa5e93e2884ba504733b9d8194ef5c5a04c1`.
The exact embedded portable plan is committed as `clean-plan.json`; the raw
root plan contained a host path and is bound only by digest.

## Measured runtime evidence

The real-model pilot used LoRA adapters on all seven declared projection
families. Phase one reached step 1 with loss `4.794458866119385`; phase two
resumed that checkpoint and reached step 2 with loss `1.819174885749817`.
Both phases reported the same 4,884,480-parameter, 420-tensor adapter census,
210 complete target instances, zero unexpected trainables, and finite values.
The pilot measured 62,574,708 checkpoint bytes and 23,123,131 export bytes.

The full run reported train loss `4.159115632375081`, evaluation loss
`3.0737040042877197`, a 384,180,224-byte allocated CUDA peak, and a
440,401,920-byte reserved CUDA peak. The trainer-reported 1.8402-second
runtime is retained as diagnostic telemetry only; it is not an end-to-end
benchmark or throughput claim.

The final `adapter_model.safetensors` is 19,593,064 bytes with SHA-256
`fd3eb151acf70ab072eb8a60186df782370fa182b74dd92f8630591ba7a9dba5`.
Verification was structural: Aptus recomputed the complete exported file tree
and required the PEFT adapter configuration and weight file. This packet does
not claim a fresh-process semantic generation or deployment check for the CUDA
adapter.

## Preliminary nonqualifying rehearsal

Before the qualifying sequence, a separate preliminary state completed four
gate jobs through `pilot-pass`. No full-train job was created and no full
training process launched. Installing the exact Aptus wheel into the accepted
training interpreter changed the runtime distribution closure and therefore
changed the environment binding. Aptus correctly treated the earlier pilot as
stale.

No runtime evidence or job state was carried forward. The qualifying sequence
used a byte-identical second compilation, the final environment binding, and a
fresh state containing exactly the five jobs listed above. The rehearsal is
classified as nonqualifying in `acceptance-summary.json` and bound by raw
record digests without being mixed into the accepted workflow.

## Records and retention

- `acceptance-summary.json` is the semantic rollup and claim boundary.
- `acceptance-procedure.json` records the bounded execution and independent
  verification rules.
- `runs/run-1/run-summary.json` is the sanitized projection of the final
  five-job workflow.
- `bundle-manifest.json`, `clean-plan.json`, and
  `model-policy-snapshot.v1.json` retain exact portable compiler artifacts.
- `provider-inspection.json` and `inspection-receipt.json` retain provider and
  compatibility provenance for the immutable model revision.
- `host-hardware.json`, `runtime-environment.json`, `python-packages.txt`, and
  `model-files.sha256` bind the accepted host/runtime/model inventory without
  publishing local paths or host identity. Two upstream-generated TensorBoard
  event names in the model inventory are replaced with stable logical labels;
  the raw manifest digest remains recorded separately.
- `raw-artifact-digests.json` binds the excluded raw records, logs, metrics,
  reports, archives, and binaries.
- `SHA256SUMS` covers every committed packet file other than itself.

The exact checkout was clean at handoff. No CUDA compute process or Aptus
host-global GPU lease remained. Raw job state, logs, model files, checkpoints,
adapter binaries, ZIP archives, absolute paths, process identifiers, network
identifiers, and credentials are not committed.

## Evidence boundary

This packet qualifies one execution only: the exact source commit and tree,
Ubuntu/RTX 3050 host binding, Python/CUDA package closure, immutable SmolLM2
revision, synthetic dataset digest, plan, candidate, policy snapshot, bundle
fingerprint, and five-job sequence recorded here. It does not establish
repeatability; general SmolLM2, Llama, RTX 3050, or CUDA compatibility; DDP,
FSDP, Full, int8-LoRA, QLoRA, other devices, or other environments; full-run
resume; semantic adapter reload; model quality, safety, convergence, or
training-rights conclusions; performance, throughput, cost, or guaranteed
memory fit; production readiness; or release readiness.
