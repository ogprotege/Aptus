# CUDA Campaign Phase 2A Tooling Contract

> **Status:** Phase 2A source tooling and Phase 2B sanitized recovery publication complete and independently reviewed | **Authority:** Active implementation and review reference for the opt-in tooling and bounded recovery publication; not a product-capability claim, target-runtime result, or measured-run record | **Applies to:** Aptus 0.2 CUDA campaign evidence-capture and recovery-publication work | **Audience:** Implementers, maintainers, custodians, and independent reviewers | **Owner:** CUDA runtime and release evidence | **Last reviewed:** 2026-08-09 | **Review by:** Before Phase 3, after any capture, custody, sanitizer, or publication contract change, or by 2026-09-08

Phase 2A freezes the source interfaces for opt-in capture, Phase 4 authority,
admission and activation, outcome classification, custody, sanitization,
eligibility, and publication. Independent adversarial reviews found and closed
defects in authority construction, native-outcome handling, receipt current
state, sanitizer-to-artifact cross-binding, durable prior review,
candidate-versus-publish separation, pinned receipt and inventory identity,
typed external evidence, CLI recovery, and publication rollback. Each closure
has focused regression coverage, and the integrated source gates passed on the
final stable tree as recorded below.

No Ubuntu command, model download, GPU workload, or new empirical run occurred
while building or reviewing Phase 2A. The Phase 0 private recovery facts
and the immutable 2026-08-06 acceptance packet are unchanged.

Phase 2B completed on 2026-08-09 from the exact merged Phase 2A source and only
against protected Phase 0 copies. Its [dated sanitized recovery
supplement](evidence/2026-08-09-cuda-phase0-recovery-supplement/README.md)
passed independent traceability and privacy review, finalized-byte
reverification, custody and retention checks, publication eligibility, and the
two-pass publication transaction. The intended Ubuntu host remained untouched.
Phase 3 candidate selection and measurement-control work is still pending, so
no new qualifying campaign run is authorized.

## Claim boundary

The local RTX 3050 cohort is intended to prove that Aptus can operate under the
campaign's fail-closed controls, reproducibly, and comprehensively on one exact
user-owned Ubuntu host. A fit, performance, admission, refusal, resource, or
stability boundary observed there applies only to that host and exact
configuration. It is not Aptus's global ceiling and cannot establish cloud
single-GPU portability, multi-GPU scaling, DDP, FSDP, or behavior on larger
accelerators. Those claims require separate evidence cohorts on the claimed
hardware.

Phase 2A supplies no new RTX 3050 measurement. Development tests are source
and contract evidence only. Implementing the Phase 4 authority contract does
not mean campaign Phase 4 ran: no production Phase 4 source-freeze artifact or
new target-host result was collected.

## Frozen Phase 2A source surfaces

| Surface | Responsibility and review boundary |
| --- | --- |
| `tools/cuda_campaign/contracts.py` | Canonical JSON/JSONL, exact record schemas, typed identities, event-ledger validation, deterministic IDs, and bounded reason vocabularies |
| `tools/cuda_campaign/phase4.py` | A separately sealed source-freeze record, exactly 600 canonical 1 Hz idle samples, and a seal; clean Git commit/tree plus current Linux/NVIDIA host, boot, journal, telemetry, and thermal bindings; distinct live and retained verification paths |
| `tools/cuda_campaign/admission.py` | Content-bound resource budgets; 120 consecutive 1 Hz observations within a 1,800-second acquisition ceiling; current Phase 4 authority; no execution/run identity before admission; and sealed activation afterward |
| `tools/cuda_campaign/monitoring.py` | Linux/NVIDIA probing, exact byte normalization, fixed-rate sampling, safety-state evaluation, summaries, idle-baseline validation, and baseline-relative cooldown validation |
| `tools/cuda_campaign/sidecar.py` | Background telemetry and an independent watchdog. Direct or injected construction remains nonqualifying; only the harness-authorized production path can hold qualifying authority |
| `tools/cuda_campaign/runtime_events.py`, `src/aptus/execution.py`, and generated CUDA `campaign_events.py`, `preflight.py`, and `train.py` | Identity-pinned, append-only pilot/train boundary journals, monotonic child-process timing, owned cancellation, parent verification, canonical records, and safe verified-byte capture; ordinary noncampaign execution remains available |
| `tools/cuda_campaign/outcomes.py` | All seven native outcomes, independent evidence status, exact stopping prefixes, cancellation-chain consistency, and publication qualification limited to `passed` intersected with `protocol-valid` |
| `tools/cuda_campaign/qualification.py` | Frozen five-action qualification, exact role inventory, telemetry/cooldown decisions, selected-artifact digests, and terminal attempt/run records |
| `tools/cuda_campaign/harness.py` | Exact-argv command capture, managed-job supervision, the qualifying managed sequence, event-ledger construction, telemetry lifecycle, owned cancellation, fallback capture, and sealed artifact assembly |
| `tools/cuda_campaign/storage.py` | Private no-clobber payload storage, semantic sealing, deep verification, append-only receipts, byte-identical copy verification, full restore, retention state, and immutable capture-failure artifacts |
| `tools/cuda_campaign/sanitizer.py` | Verified recovery-context loading, allowlisted projection, a durable sealed prior review, a distinct nonpublished finalization packet, three-role procedural separation, and finalized-candidate reverification |
| `tools/cuda_campaign/eligibility.py` | Read-only, fail-closed publication eligibility across the exact sealed publication candidate, sealed artifact, pinned custody chain, external recovery attestation, five referenced evidence files, retention, sanitizer finalization, and independent review |
| `tools/cuda_campaign/publication.py` | Content-addressed operational publication decisions, reviewed-byte allowlisting, two same-invocation live eligibility passes with internal clocks, exact inventory and receipt-head rebinding, inode-pinned staging, atomic no-replace commit, and verified rollback on post-commit failure |
| `tools/cuda_campaign/cli.py` | Privacy-bounded operator entrypoints for probing, nonqualifying command capture, intent-journaled copy/retrieval custody, distinct review/finalize/seal/evaluate/publish transitions, and crash reconciliation |

The managed-sequence contract requires the exact action order
`dependency`, `model-data`, `preflight`, `pilot`, `train`. A passing sequence
verifies exactly two retained runtime journals: four ordered pilot boundaries
and six ordered train/export/parent-verification boundaries. A protocol-valid
nonpassing sequence instead verifies the exact applicable started-action and
runtime-boundary prefix at its frozen stopping point. In both cases, canonical
records, job IDs, actions, phases, timestamps, outcomes, and reason codes must
equal the emitted runtime rows in the sealed event ledger.

A passing qualifying seal requires exactly one digest-bound copy of each of
seven frozen output roles: `plan`, `bundle-manifest`, `validation-report`,
`pilot-metrics`, `training-metrics`, `final-export-manifest`, and
`bundle-archive`. The raw artifact retains and deeply verifies the Phase 4
source-freeze record, seal, and 600-sample JSONL. Retained-byte verification is
separate from the production factory's live re-verification of current source,
host, boot, journal, telemetry, and thermal authority.

A production qualifying raw artifact also retains the planned-slot context
plus all seven activation files: six canonical activation records and
`ACTIVATED.json`. Their exact paths, roles, digests, identities, and semantic
chain are raw-manifest-bound and deeply reverified from retained bytes.
Omission, substitution, tampering, unsafe links, or inode replacement fails
closed.

## Operator command surface

The parser can be inspected without touching protected evidence:

```bash
PYTHONPATH=src:. .venv/bin/python -m tools.cuda_campaign --help
```

Do not use these mutating commands against protected evidence until Phase 2A
has merged and the Phase 2B procedure pins that merged source. The source
contract separates these transitions:

1. `sanitize-recovery-stage` writes only a protected projection stage;
2. `review-recovery-stage` performs a read-only review, while
   `seal-projection-review` creates its distinct durable sealed artifact;
3. `finalize-publication-candidate` consumes that prior review and creates a
   protected **nonpublished** packet; `verify-finalized-candidate` only
   reverifies it;
4. `seal-publication-candidate` binds the exact artifact, receipt chain,
   external evidence, sanitizer packet, campaign, and claim;
5. `verify-publication-candidate` is read-only, and
   `evaluate-publication` remains a read-only eligibility decision; and
6. `publish-candidate` performs two fresh live eligibility passes with internal
   real-time clocks, snapshots reviewed bytes, creates no eligible decision
   anchor until the final pass succeeds, pins the verified staging directory,
   commits with atomic no-replace semantics, and verifies after rename. Failed
   post-commit verification or parent `fsync` rolls the directory out of the
   public destination and verifies that destination is absent. Neither the
   production API nor CLI exposes a caller-controlled publication clock.

Copy and retrieval mutations require an explicit `operation_id` and durable
intent before mutation. `resume-operation` can reconcile a byte-identical copy
without copying it again. A retrieval outcome that was not durably recorded
cannot reconstruct its monotonic duration from wall time: it fails closed and
requires a new independently timed retrieval with a fresh operation ID and
destination. If the receipt chain moves beyond the intent-pinned head before
its exact receipt exists, resume refuses rather than silently changing
chronology.

The legacy finalize-as-public and dictionary sanitizer paths remain absent.
The earlier stage-review-finalize sequence is not frozen authority.

## Phase 2A verification gates

Run the campaign-specific source gates from the repository root:

```bash
PYTHONPATH=src:. .venv/bin/python -m unittest discover -s tests/tools -p 'test_cuda_campaign_*.py' -v
.venv/bin/ruff check tools/cuda_campaign tests/tools/test_cuda_campaign_*.py
.venv/bin/ruff format --check tools/cuda_campaign tests/tools/test_cuda_campaign_*.py
```

The exact integrated results below are from the final stable-tree closeout.
They are local development-Mac source gates, not Ubuntu, CUDA-runtime,
GPU-resource, thermal, timing, model, or empirical campaign results.

- all 302 CUDA campaign source tests passed in 22.833 seconds;
- Ruff lint and formatting checks passed for the complete campaign source and
  test surface; and
- Python bytecode compilation and the repository diff-integrity check passed.

The encompassing repository closeout passed all 888 Python tests in 45.777
seconds, all 130 React tests, generated-contract and version checks, TypeScript,
the tracked production web build, installed-wheel packaged-workbench smoke,
native macOS tests, app verification, and DMG creation. Those broader gates
remain development-host compatibility and packaging evidence only.

The adversarial coverage includes success, nonzero exit, timeout, telemetry
and watchdog failure, ownership uncertainty, cancellation milestones, partial
or malformed evidence, canonical-byte and identity mutation, symlinks and
hardlinks, path escape, interrupted sealing, duplicate run IDs, copy/retrieval
mismatch, runtime-journal omission/insertion/reorder/mutation, selected-role
omission/substitution, idle-baseline drift, sanitizer inventory defects, and
publication custody or attestation defects.

## Trust and custody boundaries

- Protected directories must already be private and must remain outside Git.
  The tooling uses no-clobber writes, rejects unsafe links and path overlap,
  and verifies sealed bytes before custody or publication decisions.
- A protocol-valid primitive `command` or `managed-job` capture is not a
  qualifying managed sequence and is rejected for publication eligibility.
- A protocol-valid managed sequence is still ineligible unless its retained
  sequence summary and every started action have `native_outcome=passed`,
  `reason_code=NONE`, and an exact successful terminal disposition.
- Publication requires two current verified copies in distinct failure
  domains, a current successful off-host retrieval, current retention and
  renewal state, and an independently bound external recovery attestation.
- The five external attestation references must be distinct and must map to
  actual single-link regular files whose SHA-256 digests match the attestation.
  Reference IDs or syntactically valid digest strings alone are insufficient.
- Producer, reviewer, and finalizer role IDs use one exact bounded procedural
  grammar and must be distinct. They record procedural separation inside the
  single-user trust boundary; they are not cryptographic authentication. A
  passing review boolean or an unsealed projection is insufficient.
- Publication eligibility is read-only. It does not copy evidence, repair a
  receipt chain, sanitize records, or broaden a claim.
- Publication is a separate mutation. An older decision never authorizes new
  bytes: the publisher holds the receipt-chain transaction, runs two fresh live
  eligibility passes, rejects candidate, inventory, and receipt-tail movement,
  and does not create an eligible anchor before the final pass. It pins the
  staging directory through no-replace commit and removes the public
  destination if post-commit verification or parent durability fails.
- Raw logs, job state, host or user identity, network addresses, private
  locators, secrets, weights, checkpoints, and adapters remain protected.
- The Phase 2 tooling does not implement Phase 3 candidate selection, add
  cloud orchestration, or create multi-GPU evidence.

## Phase 2B completion

Phase 2B pinned merged Phase 2A commit
`f6a58612263ccd1b7284ffa9f5460631ba64c2e1`, verified both immutable Phase 0
copies, and created a new canonical sealed recovery artifact without changing
the originals. Thirty-nine of 40 logical expected rows were recovered and
digest-matching. The original raw model-file manifest remained `not-found`.
The separate original Python test transcript also remained `not-found` and was
not reconstructed.

The canonical artifact was copied into two verified physical failure domains,
including an encrypted removable off-host store, and a fresh full retrieval
verified the complete sealed inventory. Typed copy, retrieval, retention, and
external-evidence records bound the exact artifact. The constructive sanitizer
then produced a protected stage; the independent review passed all six frozen
checks; a distinct finalizer created and reverified the nonpublished candidate;
and publication eligibility returned `eligible: true` with no reason codes.
The publisher performed two fresh live eligibility passes before committing
the exact reviewed bytes.

The resulting [dated checksum-covered recovery
supplement](evidence/2026-08-09-cuda-phase0-recovery-supplement/README.md) is
recovery-integrity evidence only. No Linux connection, Ubuntu-host mutation,
model download, GPU workload, or new empirical run occurred during Phase 2B.
Phase 3 is a separate gate, and neither Phase 2A nor Phase 2B authorizes a
measured run by itself.

## Related documentation

- [RTX 3050 CUDA empirical evidence campaign](cuda-empirical-campaign.md)
- [Frozen Phase 1 CUDA campaign protocol](../reference/cuda-campaign-protocol.md)
- [CUDA campaign protocol machine companion](../reference/cuda-campaign-protocol.v1.json)
- [State, storage, and retention](state-storage-retention.md)
- [Release gates](release-gates.md)
- [Release evidence template](release-evidence-template.md)
- [Documentation debt, DOC-011](../maintenance/documentation-debt.md#doc-011-publish-versioned-target-host-release-evidence)
