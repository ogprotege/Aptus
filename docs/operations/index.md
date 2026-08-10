# Operations

> **Status:** Active | **Authority:** Documentation navigation | **Applies to:** Aptus 0.2 | **Audience:** Operators and release maintainers | **Last reviewed:** 2026-08-10 | **Review by:** 2026-10-27 or when operational behavior changes

## Run a bundle

- [Operator checklist](operator-checklist.md)
- [Compile, validate, and run](../guides/compile-validate-run.md)
- [State, storage, and retention](state-storage-retention.md)
- [Recovery and resume boundary](../guides/resume-recover.md)
- [Security policy](../../SECURITY.md)

## Prepare a release

- [Release gates](release-gates.md)
- [Release evidence template](release-evidence-template.md)
- [2026-08-10 CUDA Phase 6 Full confirmatory stability](evidence/2026-08-10-cuda-phase6-confirmatory-stability/README.md)
- [2026-08-10 CUDA Phase 6 remediation method matrix](evidence/2026-08-10-cuda-phase6-remediation-matrix/README.md)
- [2026-08-10 CUDA Phase 5 repeatability anchor](evidence/2026-08-10-cuda-phase5-repeatability-anchor/README.md)
- [2026-08-10 historical CUDA Phase 6 same-model method matrix](evidence/2026-08-10-cuda-phase6-method-matrix/README.md)
- [2026-08-10 CUDA Phase 5 retention addendum](evidence/2026-08-10-cuda-phase5-repeatability-retention.json)
- [2026-08-09 Phase 2B sanitized Phase 0 recovery supplement](evidence/2026-08-09-cuda-phase0-recovery-supplement/README.md)
- [2026-08-06 SmolLM2 CUDA LoRA single-device acceptance](evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/README.md)
- [2026-08-05 Phase 6 Qwen2 MLX-LM exact-source acceptance](evidence/2026-08-05-qwen2-mlx-lm-exact-source-refresh/README.md)
- [2026-08-05 original Phase 6 Qwen2 MLX-LM acceptance baseline](evidence/2026-08-05-qwen2-mlx-lm-acceptance/README.md)
- [2026-07-27 MLX-LM target-host acceptance](evidence/2026-07-27-mlx-lm-acceptance/README.md)
- [2026-07-27 desktop engineering acceptance](evidence/2026-07-27-desktop-release/README.md)
- [Changelog](../../CHANGELOG.md)

### Complete Ubuntu CUDA acceptance packet

The repository retains a checksum-covered, sanitized 15-file record for the
2026-08-06 Ubuntu 24.04.4 and NVIDIA RTX 3050 acceptance. Start with the
immutable narrative in this order:

1. [Result](evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/README.md#result)
2. [Bound inputs](evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/README.md#bound-inputs)
3. [Source, runtime, and compilation](evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/README.md#source-runtime-and-compilation)
4. [Measured runtime evidence](evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/README.md#measured-runtime-evidence)
5. [Preliminary nonqualifying rehearsal](evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/README.md#preliminary-nonqualifying-rehearsal)
6. [Records and retention](evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/README.md#records-and-retention)
7. [Evidence boundary](evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/README.md#evidence-boundary)

Every committed packet file is indexed below. Together, these are the entire
repository-retained result set for that execution.

| Evidence layer | Complete committed records |
| --- | --- |
| Narrative and semantic result | [`README.md`](evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/README.md), [`acceptance-summary.json`](evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/acceptance-summary.json) |
| Procedure and five-job projection | [`acceptance-procedure.json`](evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/acceptance-procedure.json), [`runs/run-1/run-summary.json`](evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/runs/run-1/run-summary.json) |
| Compiler input and policy | [`bundle-manifest.json`](evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/bundle-manifest.json), [`clean-plan.json`](evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/clean-plan.json), [`model-policy-snapshot.v1.json`](evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/model-policy-snapshot.v1.json) |
| Model provenance | [`provider-inspection.json`](evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/provider-inspection.json), [`inspection-receipt.json`](evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/inspection-receipt.json), [`model-files.sha256`](evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/model-files.sha256) |
| Host and runtime | [`host-hardware.json`](evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/host-hardware.json), [`runtime-environment.json`](evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/runtime-environment.json), [`python-packages.txt`](evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/python-packages.txt) |
| Excluded-raw bindings and packet integrity | [`raw-artifact-digests.json`](evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/raw-artifact-digests.json), [`SHA256SUMS`](evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/SHA256SUMS) |

The exact source summary records that 550 Python tests passed in 37.712
seconds. The detailed sanitized runtime results are in `run-summary.json`,
including all five jobs, timestamps, return codes, bindings, preflight and
pilot metrics, full-training metrics, export manifest, parent promotion, and
handoff state.

Critical retention boundary: the repository does **not** contain the original
verbose Python test stdout/stderr, raw job records, raw per-job logs, model or
adapter binaries, or bundle archives. The packet deliberately marks those
items as uncommitted, and `raw-artifact-digests.json` binds each excluded job
record and job log by SHA-256. It does not record a digest or durable external
location for the Python test transcript itself. The raw artifacts were marked
as retained outside the repository when the packet was produced, but their
external location is not part of the committed record. Consequently, the
packet proves the recorded summary and structured results; it must not be
described as a publication of the complete raw console output.

Private Phase 0 recovery completed on 2026-08-08 without changing that public
packet or publishing protected locations, host identities, job identities, or
raw content. On 2026-08-09, Phase 2B published the independently reviewed
[sanitized recovery supplement](evidence/2026-08-09-cuda-phase0-recovery-supplement/README.md):
39 of 40 frozen logical rows were recovered with matching digests, the raw
model-file manifest was not found, and the separately searched Python test
transcript was not found or reconstructed. No new Ubuntu run occurred.

## Experimental host work

- [RTX 3050 CUDA empirical evidence campaign](cuda-empirical-campaign.md)
- [Phase 2 tooling and completed Phase 2B publication contract](cuda-campaign-phase2-tooling.md)
- [Frozen Phase 1 CUDA campaign protocol](../reference/cuda-campaign-protocol.md)
- [CUDA campaign protocol machine companion](../reference/cuda-campaign-protocol.v1.json)
- [Apple Silicon fine-tuning experiment matrix](apple-silicon-pilot.md)

The CUDA campaign is the canonical ordered plan for recovering the existing raw
Ubuntu records, implementing complete capture and explicit candidate selection,
then measuring single-device repeatability, method behavior, scale, guarded
capacity frontiers, and endurance. It does not itself assert a result, and one
RTX 3050 cannot supply DDP or FSDP evidence.

Phase 0 is privately complete. The Phase 1 protocol is frozen design authority.
Phase 2A implements and independently reviews the opt-in Phase 4 authority,
admission/activation, outcomes, capture, telemetry, watchdog, custody,
sanitizer, eligibility, and publication source interfaces, with its integrated
source gates complete. It is not operator authorization
or target-runtime evidence. Phase 2B used that merged source to publish and
independently review the sanitized recovery supplement without connecting to or
mutating the Ubuntu host. Phase 3 explicit selection and measurement controls,
Phase 4 rehearsal and freeze, the successful five-slot Phase 5 repeatability
anchor, and Phase 6 are complete. The immutable Phase 6 remediation packet
records Full promotion followed by one pass and four unrelated-GPU-activity
safety cancellations. After two Aptus-owned pre-launch registration races were
corrected, a separate five-slot Full cohort passed the frozen stability and
integrity contract at exact merged source `2bc4d9a`. One stable Full cell is
therefore established and Phase 7 is authorized. The earlier Phase 6 packets
and intervening source-defect diagnostic remain separate historical evidence.

The Apple matrix combines current v5/v3 small-model MLX-LM QLoRA acceptance
at exact source `719255153e3fc7e38e83b5ff826d587e5e58bf80`, the original Phase 6
baseline, historical v2/v2 evidence, and proposed larger-model work. The
separate record above covers one exact 2026-08-06 CUDA LoRA single-device
acceptance and identifies the remaining CUDA target-host work. Every result is
scoped to its exact artifact, source tree,
bundle fingerprint, host, and environment; none transfers to another runtime
path or establishes repeatability, safety, quality, performance, production
readiness, or release readiness. Read the linked immutable evidence record
before treating a row as passed.

## Related documentation

- [Documentation home](../index.md)
- [RTX 3050 CUDA empirical evidence campaign](cuda-empirical-campaign.md)
- [Phase 2 tooling and Phase 2B completion](cuda-campaign-phase2-tooling.md)
- [CUDA campaign protocol](../reference/cuda-campaign-protocol.md)
- [Run states](../reference/run-states.md)
- [Troubleshooting](../guides/troubleshooting.md)
