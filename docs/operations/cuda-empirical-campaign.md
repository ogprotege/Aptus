# RTX 3050 CUDA Empirical Evidence Campaign

> **Status:** Active experiment plan; Phases 0 through 6 complete; Phase 6 produced no promoted method and Phase 7 is not authorized | **Authority:** Canonical operational plan for bounded CUDA evidence; non-normative for current capability | **Applies to:** Aptus 0.2 single-device CUDA characterization on the intended Ubuntu RTX 3050 host | **Audience:** Operators, maintainers, and evidence reviewers | **Owner:** CUDA runtime and release evidence | **Last reviewed:** 2026-08-10 | **Review by:** Before authorizing a replacement cohort, changing the capture contract, or by 2026-09-09

This is the one execution plan for the next CUDA evidence campaign. It combines
the remaining roadmap work, release gates, evidence-packet requirements,
storage rules, operator procedure, calibration methodology, and fair-comparison
guidance into one ordered program without taking authority away from those
documents.

The plan schedules work; it does not assert that a scheduled run passed. Only a
reviewed, dated, checksum-covered record under [`evidence/`](evidence/) can
establish a measured result. Current capability language remains unchanged
until such a record passes independent review.

As of 2026-08-09, Phase 0 recovery is complete in the protected private layer:
the expected prior evidence has a private disposition, two verified copies
exist in separate failure domains, and off-host retrieval passed. No private
path, machine identity, job identity, or raw record is published by this status
note. Phase 2B published and independently reviewed the [sanitized recovery
supplement](evidence/2026-08-09-cuda-phase0-recovery-supplement/README.md)
from those protected copies without connecting to the Ubuntu host. The
[Phase 1 CUDA campaign protocol](../reference/cuda-campaign-protocol.md) and its
[machine-readable companion](../reference/cuda-campaign-protocol.v1.json) are
the frozen design authority. The [Phase 2A tooling
contract](cuda-campaign-phase2-tooling.md) records the implemented source
interfaces, closed review findings, and completed Phase 2B publication; it
records no new Ubuntu or GPU result. Phase 3 was implemented locally without a
Linux connection or campaign run.

## What this plan reconciles

| Existing document | Existing authority | How this campaign uses it |
| --- | --- | --- |
| [`ROADMAP.md`](../../ROADMAP.md) | Product sequence and release scope | Tracks this campaign as the single-device CUDA workstream; it does not contain the runbook |
| [Documentation debt, DOC-011](../maintenance/documentation-debt.md#doc-011-publish-versioned-target-host-release-evidence) | Open evidence debt | Tracks which campaign outcomes remain missing |
| [Release gates](release-gates.md) | Normative acceptance requirements | Every qualifying run and publication must satisfy the applicable gates |
| [Release evidence template](release-evidence-template.md) | Public packet structure | Each result batch instantiates the template; this plan does not invent a second packet format |
| [Operator checklist](operator-checklist.md) | One-bundle execution procedure | Every run follows its five ordered actions and state-handling rules |
| [State, storage, and retention](state-storage-retention.md) | Storage and retention boundary | Raw records go to a protected non-Git vault; only sanitized projections enter Git |
| [Preflight and calibration](../methodology/preflight-calibration.md) | Normative measurement interpretation | One host is a characterization cohort, not statistical calibration |
| [Design an evaluation](../guides/design-an-evaluation.md) | Fair quality-comparison controls | Resource comparisons use its controls; quality claims require a separate frozen evaluation |
| [Method registry](../reference/method-registry.md) | Executable method, runtime, backend, and placement matrix | Defines which CUDA cells may be attempted |
| [Model-policy snapshot](../reference/model-policy-snapshot.md) | Canonical model-policy decision contract | Defines whether an inspected artifact can produce the intended CUDA candidate |
| [CUDA campaign protocol](../reference/cuda-campaign-protocol.md) and [machine companion](../reference/cuda-campaign-protocol.v1.json) | Frozen Phase 1 experiment contract | Defines canonical identities, fixtures, thresholds, schedules, aggregation, retention, sanitization, and stopping semantics; it does not implement them |
| [Phase 2 tooling and recovery-publication contract](cuda-campaign-phase2-tooling.md) | Implemented source and review contract | Freezes the opt-in capture, Phase 4 authority, admission/activation, outcome, custody, sanitizer, eligibility, and publication interfaces and records completed Phase 2B publication without asserting target-runtime evidence |
| [Apple Silicon pilot matrix](apple-silicon-pilot.md) | Apple/MLX experiment plan | Remains a sibling plan and supplies no CUDA evidence |
| [Capability matrix](../reference/capability-matrix.md) | Current executable and evidence status | Changes only after reviewed campaign packets exist |
| Ignored `WIP.md`, `TempDoc-ForUserReview/`, and `dev/archive/` | Local notes or historical review evidence | Useful historical lessons were incorporated here; these paths are not current authority |

If this page conflicts with a normative release gate, the release gate wins. If
it conflicts with executable code or a generated contract, stop, record the
discrepancy, and reconcile the plan before running.

## Claim boundary

The intended host is a user-operated Ubuntu workstation reported to contain
one NVIDIA GeForce RTX 3050 with 8 GiB dedicated VRAM and ample host RAM and
disk. Those are declared starting facts, not campaign measurements. Phase 0
must acquire and bind the exact operating-system, CPU, RAM, storage, GPU, GPU
UUID, driver, CUDA, thermal-limit, and runtime facts before any qualifying run.
The public record must replace private machine identifiers with an opaque host
ID.

This single-GPU campaign can characterize only these `single` CUDA cells:

- Full fine-tuning, subject to BF16 and live admission;
- LoRA;
- int8-LoRA, subject to the exact bitsandbytes kernel and environment gates; and
- QLoRA, subject to the exact bitsandbytes kernel and environment gates.

It cannot execute DDP or LoRA FSDP because both require at least two GPUs. Full
FSDP, int8-LoRA FSDP, and QLoRA FSDP remain unsupported. Distributed rows stay
open for a separately approved multi-GPU campaign; a one-GPU refusal or static
test is not distributed runtime acceptance.

The campaign may support exact-host facts about completion, elapsed time,
memory, utilization, temperature, power sampling, artifact size, and bounded
fit. It cannot by itself establish safety, model quality, production
throughput, universal fit, universal method ranking, cost, or statistical
estimator calibration.

Every boundary from this campaign is an exact-host local observation, not an
Aptus-wide ceiling. Cloud single-GPU, larger-accelerator, and multi-GPU claims
require their own evidence cohorts. A workload refused on this RTX 3050 means
only that it was not safely admitted on this exact host and configuration.

## Evidence vocabulary

- **Declared** means supplied by an operator or configuration and not observed
  by the run.
- **Inferred** means deterministically derived from bound declared or measured
  facts.
- **Estimated** means produced by a model or sampling calculation. Integrating
  sampled power is an energy estimate, not a meter reading.
- **Measured** means observed by the bound runtime or capture process on the
  exact host during the exact run.
- **Rehearsal** means a deliberately nonqualifying harness test.
- **Slot status** is either `started` when the harness activates the slot or
  `planned-not-started` when a prior frozen rule prevents activation.
- **Native execution outcome** for a started slot is `passed`, `refused`,
  `failed`, `cancelled`, `timed-out`, `guard-blocked`, or `unknown`. It records
  what execution did, independently of evidence quality.
- **Evidence status** is `protocol-valid`, `capture-invalid`, or `not-started`.
  A protocol-valid attempt is a fresh, predeclared started slot whose source,
  configuration, capture, retention, and acceptance contracts were frozen and
  completely captured. Its native execution outcome may still be non-passing.
- **Refused** means Aptus stopped at a gate before the requested action. It is
  useful negative evidence, never a pass.
- **Failed** includes nonzero exits, OOM, non-finite values, or invalid runtime
  behavior. A safety guard that prevents execution is `guard-blocked`; a guard
  that cancels active execution is `cancelled` with the trigger recorded.
  Capture failure changes evidence status, never the native outcome. Every
  planned slot stays in its applicable denominator, and no status or outcome is
  silently replaced or discarded. Only a native `passed` outcome with
  `protocol-valid` evidence may support a pass claim.

## Non-negotiable controls

1. Never edit a generated plan, selected candidate, bundle configuration, or
   evidence file to force a matrix cell. Use the Phase 3 validated selection
   contract to create a new plan identity before compiling an alternate viable
   method.
2. Never reuse a run output directory or overwrite a prior state root. Every
   repetition receives fresh state, bundle, run, and capture locations.
3. Never install, update, clean, or start Aptus on the Ubuntu host until Phase 0
   has produced two checksum-verified recovery copies of the August 6 raw
   records in separate failure domains, including one off-host, Phase 2B has
   published the reviewed sanitized recovery supplement, and retrieval has
   passed.
4. Never overclock the GPU, raise its power limit, bypass Aptus admission, or
   disable thermal protection to find a boundary.
5. Never publish raw logs, job state, usernames, hostnames, IP addresses, GPU
   UUIDs, home paths, tokens, private data, model weights, checkpoints, or
   adapters in Git.
6. Never call a run comparable after changing source, model revision, dataset,
   candidate identity, dependency closure, sequence length, effective batch,
   step or token budget, capture method, or host policy without creating a new
   comparison cell.
7. Report tokens per second only from the exact Phase 3 padded, non-padding,
   and supervised-token counters bound into the completed run. Never infer
   token throughput from configured sequence length or aggregate trainer rates.

## Campaign identifiers and immutable configuration

The capture-infrastructure change must define versioned, canonical records at
six levels:

1. **Campaign ID:** the complete bounded program.
2. **Comparison-cohort ID:** the question, held controls, varied dimensions,
   member cells, paired-seed schedule, randomized-complete-block schedule,
   attempt counts, stopping rule, and aggregate decision rule.
3. **Comparison-cell ID:** one model, method, placement, and stable
   configuration whose repetitions can be summarized together. It binds the
   seed policy but excludes the repetition's actual seed, ordinal, fresh paths,
   timestamps, and run-order slot.
4. **Attempt-slot ID:** one immutable, predeclared cohort-plan slot, binding its
   intended cell, block, ordinal, role, order position, and scheduled seed. It
   exists even when the slot never starts.
5. **Execution-configuration ID:** every behavior-affecting value for one
   attempt, including its actual seed. Two repetitions may share this ID only
   when all exact behavior values are identical. It is absent when a slot does
   not start.
6. **Experiment-run ID:** the actual invocation and execution context. It is
   assigned only when a slot starts and binds back to the attempt-slot ID.

The comparison cell and exact execution configuration must bind at least:

- full source commit, tree digest, clean/dirty status, and diff digest when a
  deliberately dirty rehearsal is allowed;
- canonical command template, path-allocation policy, and allowlisted
  environment policy;
- host and hardware binding, driver, CUDA runtime, Python, direct pins, and
  complete installed distribution closure;
- model and tokenizer repository plus immutable revision, model-file digests,
  provider-inspection receipt, and policy snapshot;
- canonical training-data digest, schema, row count, split-group rules, and
  train/validation assignment;
- method, precision, quantization, placement, world size, sequence length,
  micro-batch, accumulation, effective batch, epochs or optimizer-step budget,
  checkpoint cadence, adapter targets, seed policy, and the execution
  configuration's actual seed;
- cold/warm model-cache policy, warm-up treatment, cooldown
  rule, maximum wall time or update count, numeric stop thresholds, and
  watchdog behavior; and
- capture-tool version, sample interval, clock source, sanitization rules, and
  raw-retention policy, including p95 interpolation and telemetry-coverage
  calculation.

The experiment-run record binds the exact argv, working directory, fresh state
root, bundle and output paths, timestamps, run-order block and slot, observed
host state, environment binding, plan ID, candidate ID, bundle fingerprint,
bundle-manifest digest, archive SHA-256, every Aptus job and full-run ID, and all
terminal evidence. Aptus has no generic `bundle_id`; do not invent one.

Changing a held comparison factor creates a new comparison cell. Changing an
exact behavior value such as the actual seed creates a new execution
configuration within that cell. Activating a slot assigns fresh run IDs, paths,
timestamps, and execution context without breaking the cell grouping. A slot
that never starts keeps its attempt-slot ID but has no execution-configuration
or experiment-run ID. Aggregation is by predeclared comparison cell; paired
method comparisons additionally bind the comparison cohort and complete-block
slot.

## Raw and public evidence layers

### Protected raw vault

Before a qualifying run, select a non-Git vault path and record its custodian,
retention-policy ID, and provisional retention not-before date. Maintain two
checksum-verified copies in separate failure domains, with at least one off the
Ubuntu host, and test retrieval from the second copy. On POSIX, directories use
mode `0700` and files use mode `0600`. Any copy leaving the trusted local
filesystem requires equivalent access control, encryption in transit and at
rest, and a recorded key custodian and recovery procedure. Each terminal run is
sealed into a no-clobber directory containing:

- exact argv, working directory, start and finish timestamps, monotonic and
  wall duration, exit code, source identity, and allowlisted environment;
- byte-exact combined stdout/stderr in observed order, or separate streams when
  ordering cannot be preserved, with SHA-256 and byte size;
- the complete Aptus job JSON and job log for every action, plus report,
  metrics, manifest, checkpoint, export, and completion bindings;
- raw GPU and host telemetry, its schema, interval, timestamp basis, missed
  samples, and collection-process result;
- an inventory of every raw file with relative path, media type, size, and
  SHA-256; and
- a sealed manifest, retention-policy ID, provisional retention not-before date,
  and opaque protected artifact ID.

To **seal** a run, write a versioned canonical manifest to a fresh directory,
flush every file, create the completion marker atomically without clobbering,
hash the completed manifest, and anchor that hash in both the off-host copy and
the sanitized public packet. Any later content change must fail verification;
never reseal a changed directory under the same experiment-run ID.

Retrieval, copy verification, publication, retention renewal, and claim
withdrawal are append-only receipts that reference the immutable raw-manifest
digest; they never modify the sealed run. At publication, issue a retention
receipt whose effective date is at least 24 months after packet merge and until
the v0.2 claim is withdrawn or superseded, whichever is later. Verify both
copies at the Phase 1-frozen cadence, after any storage, key, or custodian
change, and at least 90 days before expiry. Renew the receipt or withdraw the
claim before removal. Loss of a required copy or failed retrieval makes the
claim nonqualifying until redundancy and retrieval are restored and reviewed.
If consent, license, security, or another controlling requirement forces
earlier deletion, publish a withdrawal record and mark the claim nonqualifying
before deletion whenever permitted; the retention policy never overrides such
a requirement. Other legal, incident, or research rules may require longer
retention.

The raw vault is the audit source. Aptus's API log tail and a public summary are
not substitutes for the complete job log.

### Sanitized Git packet

Each completed batch gets a new dated, immutable directory under
[`docs/operations/evidence/`](evidence/). It contains bounded numeric summaries,
slot, configuration, and run ledgers, failure and refusal rows, stable reason
codes with bounded sanitized explanations, exact source/model/data bindings,
aggregate calculations, public-safe host facts, raw-to-sanitized digest
mappings, canonical raw-manifest digests and byte sizes, protected opaque
locators, append-only retention-receipt bindings, two-copy verification,
retrieval dates/results, known limitations, and `SHA256SUMS`. Byte-exact
exception text stays only in the protected vault and is referenced publicly by
digest; it is never copied into the sanitized reason.

The sanitizer must use an allowlist. A reviewer must verify both that private
values are absent and that every published number can be traced to a sealed raw
record. Existing evidence packets, including the August 6 CUDA packet, remain
immutable.

## Required measurement channels

### Time

Use one monotonic event ledger with a recorded UTC wall-clock mapping. Emit
markers for harness start, every command start/finish, every managed job-state
transition, both pilot phases, training, export, parent verification, telemetry
start/stop, sealing, and cooldown. Segment telemetry and sampled GPU-energy
estimates by those same markers so download, pilot, training, and verification
cannot be conflated.

Keep these durations separate:

- dependency, model-data, preflight, pilot phase 1, pilot phase 2, training,
  export, and parent-verification duration;
- trainer-reported runtime and externally measured child-process runtime;
- managed-job queued-to-terminal duration;
- end-to-end five-action repetition duration, excluding separately captured
  environment provisioning and cooldown; and
- clean environment installation and model download time, each captured as a
  separate setup record with its own command, transcript, and event boundaries.

If environment installation or model download is not captured from before its
command starts, exclude that duration from every performance aggregate rather
than reconstructing it later.

### Accelerator and host resources

Sample at a predeclared interval between 0.5 Hz and 1 Hz. Record unsupported
fields explicitly rather than dropping them. The private stream should include,
when the driver exposes them:

- GPU UUID, memory used/total, utilization, temperature, power draw and limit,
  graphics and memory clocks, performance state, throttle reasons, and a bound
  kernel/journal projection for NVIDIA Xid events;
- host available RAM, swap use and activity, load, filesystem free bytes,
  process RSS, CPU and I/O counters, disk-growth rate, supported CPU/package and
  NVMe temperatures, and the capture process's health; and
- unrelated GPU processes at admission and any detected during execution.

Retain the runtime's own allocated and reserved CUDA peaks separately from
device-level telemetry. Publish maximum, median, and p95 telemetry observations
with sample count and coverage, using the interpolation and coverage formulas
frozen before the run. If sampled GPU power is integrated over time, publish
the numerical method and interval and label the result **estimated GPU
energy**, never host or whole-machine energy.

### Training and artifacts

Record completed optimizer steps, finite losses, exact trainable-parameter
census and digest, dataset split counts and digests, checkpoint-continuation
proof, structural export result, checkpoint/export sizes, and failure state.
Do not infer task quality from loss, completion, or structural export.

## Guarded host protocol

Before each run:

1. verify the exact frozen configuration and free no-clobber paths;
2. verify no unrelated GPU workload, Xid error, active thermal throttle, or
   unexpected clock/power setting;
3. record idle GPU and supported CPU/NVMe temperatures, free VRAM, host RAM,
   swap, load, free disk, and current disk/vault budgets;
4. start capture before the first Aptus command and prove its clock is live;
5. run dependency, model-data, measured-preflight, pilot, and confirmed training
   in order, stopping at the first non-passing gate; and
6. after terminal state, stop capture, seal the vault record, sanitize a
   projection, verify checksums, and perform a test retrieval before proceeding.

Stop the current cell immediately on any of these conditions:

- Aptus admission refusal or policy replan requirement;
- CUDA OOM, Xid, applicable uncorrected hardware error, driver reset, or lost
  device;
- non-finite loss or invalid trainable, checkpoint, split, export, or completion
  evidence;
- any thermal-slowdown signal, a temperature at the warning threshold sustained
  for the predeclared warning duration, or a temperature at the numeric abort
  threshold derived from the device's reported slowdown limit; use the frozen
  conservative numeric fallback only when the slowdown limit is unavailable
  but the relevant sensor remains readable;
- free VRAM, available RAM, or free disk below its numeric reserve, including
  the filesystem hard floor and per-model download, output, and vault budgets;
- swap activity above its numeric rate and duration threshold;
- unrelated GPU activity; or
- capture-process or watchdog failure, a telemetry gap above its numeric
  duration or coverage tolerance, the predeclared maximum wall time or update
  count, or inability to seal and retrieve raw evidence.

For every live stop trigger while a managed job is active, the harness records
the trigger-detection timestamp, requests cancellation through the owning
`JobService` by exact job ID, and records the cancellation-request, confirmed
process-group termination, and global-lease reconciliation timestamps. It never
signals a stale numeric PID directly. If ownership, termination, or lease
reconciliation is uncertain, seal the incomplete attempt and block all further
submissions pending operator diagnosis.

After a safety stop, do not repeat the same failing configuration merely to
obtain another point. Diagnose first and either reduce one explicit axis or
record the bounded failure. Cool the host back to its predeclared idle band
before the next timed run.

## Ordered implementation and execution plan

The phases below are the only supported order. Phase 0 must precede host
mutation. Phase 2A is a reviewable tooling change, Phase 2B is the independently
reviewed sanitized-recovery publication, and Phase 3 is a separate selection
and measurement-control change. All applicable gates must merge before a
qualifying GPU run.

Current status is intentionally narrower than the phase descriptions: Phase 0
completed privately on 2026-08-08, Phase 1 is frozen as design authority,
Phase 2A is merged and source-gated, and Phase 2B published the independently
reviewed sanitized recovery supplement on 2026-08-09. Phase 3 now implements
the frozen selection and measurement-control prerequisites. No Ubuntu or
empirical run occurred; Phase 4 rehearsal remains the next gate.

| Phase | Outcome | May start when | Exit evidence |
| ---:| --- | --- | --- |
| 0 | Recover and protect prior Ubuntu raw evidence without publishing it yet | This plan is approved | Complete private inventory, digest comparison, two verified copies, and off-host retrieval |
| 1 | Freeze protocol, record fields, retention, fixtures, and claim boundary | Phase 0 recovery may proceed in parallel, but no host mutation occurs | Reviewed plan, recovery-supplement schema, and sanitizer decisions |
| 2A | Implement and test Phase 4 authority, admission/activation, outcomes, capture, telemetry, watchdog, custody, sanitizer, eligibility, and publication tooling | Phase 1 fields are frozen and Phase 0 inventory is complete | Source-branch campaign gates and adversarial fake/short-command evidence; no target-runtime result |
| 2B | Publish the recovered-evidence supplement through the reviewed sanitizer and custody gates | Phase 2A is reviewed and the protected Phase 0 inputs remain immutable | Dated sanitized supplement, sealed independent review, and passing publication eligibility |
| 3 | Implement explicit candidate selection and exact measurement contracts | Phase 1 selection semantics are approved | Full gates and candidate-identity mutation tests |
| 4 | Rehearse the harness and freeze the campaign source, host, environments, models, data, and run order | Phases 0, 2A, 2B, and 3 pass | Nonqualifying rehearsal, captured Ubuntu repository gates, and successful retrieval |
| 5 | Establish the LoRA 135M repeatability anchor | Phase 4 passes without protocol changes | Exactly five measured attempts and a predeclared batch decision |
| 6 | Compare all four single-device methods on the same anchor | Phase 5 is reviewed | Exploratory and predeclared measured blocks |
| 7 | Characterize model-size and then architecture breadth | Stable cells exist in Phase 6 | Staircase ledger with passes, refusals, and failures |
| 8 | Find guarded sequence and batch frontiers one axis at a time | A stable model/method cell is chosen | Largest passing and next bounded non-passing points |
| 9 | Run endurance and target-host job-control checks | A point meeting the frozen numeric headroom margins is selected | Repeated endurance and recovery records |
| 10 | Aggregate, independently review, publish, and update claims | All scheduled batches are sealed or dispositioned | Dated public packets and explicit remaining gates |

### Phase 0 — forensic recovery before host changes

1. Boot the Ubuntu host without pulling source, updating packages, running
   cleanup, starting Aptus, or reusing `.aptus-state`.
2. Locate the state root and the job IDs referenced by the August 6 packet.
3. Resolve every entry in the August 6 `raw-artifact-digests.json`: source
   bundle and wheel; root and embedded plans; bundle manifest and archive;
   policy snapshot and requirements; hardware, model, dataset, provider, and
   raw-model-file records; all qualifying and rehearsal job JSON/log pairs;
   preflight, pilot-phase, pilot, and full metrics; final export manifest;
   terminal validation report; and adapter artifact. Copy every found item plus
   any test transcript into a protected recovery area without modifying source
   files.
4. Create a second copy in a separate failure domain, with at least one copy off
   the Ubuntu host. Hash both copies, compare them byte-for-byte, and retrieve
   and verify the off-host copy before any host mutation.
5. Compare every recovered file with its expected digest in
   `raw-artifact-digests.json`; do not limit verification to jobs and logs.
6. Search for the original 550-test transcript. If it never existed as a file,
   record that fact and the search scope; do not recreate it from a summary.
7. Assign every expected raw item one result: recovered and matching, recovered
   but mismatched, or not found. Preserve the recovery inventory in the vault.
8. Preserve the private recovery inventory and its digest as Phase 2B input. Do
   not publish it with an improvised schema or sanitizer before Phase 1 review.

Phase 0 is complete only when every existing digest has a disposition, two
verified copies exist in separate failure domains, the off-host copy has passed
retrieval, and the host can remain unchanged without risking the only raw copy.
Phase 0 completion alone does not authorize host mutation.

Phase 0 met this private completion boundary on 2026-08-08. Phase 2B published
its independently reviewed [sanitized
supplement](evidence/2026-08-09-cuda-phase0-recovery-supplement/README.md) on
2026-08-09 through the reviewed sanitizer; the private inventory and locators
remain outside Git.

### Phase 1 — protocol and schema decisions

The frozen decisions are versioned in the
[human-readable protocol](../reference/cuda-campaign-protocol.md) and
[canonical machine companion](../reference/cuda-campaign-protocol.v1.json).
They define design contracts, not runtime behavior. Any discrepancy between
this scheduling page and those frozen values must be reconciled before any
measured execution.

Review and freeze:

- canonical campaign, cohort, comparison-cell, attempt-slot,
  execution-configuration, experiment-run, raw-manifest, event-ledger,
  telemetry, sanitizer, aggregate, and retrieval-proof fields;
- raw-vault path, custodian, access mode, two-copy backup, off-host encryption
  and key recovery, retention-policy ID, provisional not-before date, receipt
  cadence, campaign-specific 24-month minimum, and renewal or claim-withdrawal
  procedure;
- public redaction allowlist, stable reason-code vocabulary, bounded sanitized
  explanations, recovery-supplement schema, and independent review procedure;
- exact synthetic contract fixture and longer deterministic benchmark fixture,
  including rights and immutable digests;
- model-selection criteria and immutable-revision requirements;
- method comparison controls, exact planned slot count for every cell,
  randomized-complete-block and paired-seed schedules, warm-up policy, cache
  policy, numeric cooldown rule, promotion/stability rule, pass threshold,
  no-replacement rule, and stopping rule;
- slot-status handling for `started` and `planned-not-started`, native-outcome
  handling for `passed`, `refused`, `failed`, `cancelled`, `timed-out`,
  `guard-blocked`, and `unknown`, and independent evidence-status handling for
  `protocol-valid`, `capture-invalid`, and `not-started`, including how a
  missing block member affects paired analysis;
- GPU warning and abort temperatures relative to the reported slowdown limit
  plus a conservative numeric fallback, optional CPU/NVMe temperature limits,
  free-VRAM/RAM/disk floors, swap-rate limit, telemetry-gap tolerance, maximum
  wall time or update count, per-model storage budgets, filesystem hard floor,
  and numeric Phase 9 endurance-selection headroom above every applicable hard
  stop;
- per-run summaries, aggregate and uncertainty calculations, p95 interpolation,
  telemetry-coverage formulas, and outlier policy, all chosen before measured
  data exists;
- whether clean installation and downloads are captured as separate setup
  records or excluded from performance aggregates; and
- whether unavailable optional CPU/NVMe temperature sensors are a declared
  channel exclusion or disqualify endurance evidence; a sensor that was
  declared required or previously supported becoming unreadable invalidates
  the affected attempt; and
- which metrics are currently measurable and which require code changes.

### Phase 2A — evidence-capture infrastructure

The implemented opt-in campaign tooling under `tools/` leaves ordinary Aptus
execution available while adding fail-closed evidence authority. Its reviewed
source contract:

- retains a separately sealed three-file Phase 4 authority: source-freeze
  record, exactly 600 canonical 1 Hz idle samples, and seal, with clean source,
  exact current-host, boot, journal, telemetry, and thermal bindings;
- derives exact content-bound resource budgets, collects 120 consecutive 1 Hz
  admission observations within 1,800 seconds, and creates no execution or run
  identity until admission passes and a sealed activation is verified;
- retains the planned-slot context and all seven activation files, binding
  their paths, roles, digests, identities, and semantic chain into the raw
  manifest for deep verification;
- classifies all seven native outcomes independently of `protocol-valid`,
  `capture-invalid`, or `not-started` evidence status and accepts only exact
  stopping prefixes and cancellation chains;
- launches an exact command while capturing complete output and timing;
- emits one monotonic event ledger with UTC mapping for command, managed-job,
  pilot-phase, training, export, verification, telemetry, sealing, and cooldown
  boundaries, then derives segmented summaries from those markers;
- captures each environment-provisioning or model-download command as its own
  setup record when setup timing will be published;
- copies and seals terminal Aptus job records, complete logs, reports, metrics,
  manifests, and selected artifacts;
- runs a low-overhead NVIDIA and host telemetry sidecar;
- enforces every live safety trigger and the frozen deadline through the owning
  `JobService`: record detection, cancellation-request, confirmed process-group
  termination, and lease-reconciliation timestamps by exact job ID, then seal
  the exact terminal disposition; never signal a stale numeric PID directly,
  and block further submissions when ownership or termination is uncertain;
- requires all seven passing output roles: plan, bundle manifest, validation
  report, pilot metrics, training metrics, final-export manifest, and the
  deterministic bundle archive;
- creates no-clobber raw manifests, sanitization projections, aggregates, and
  `SHA256SUMS`;
- when normal capture or sealing cannot complete, writes an immutable
  capture-failure receipt through a protected fallback path with the
  attempt-slot and run IDs, stable failure code, available-file inventory,
  missing-field list, SHA-256, byte size, and any recoverable locator; if neither
  a canonical raw manifest nor that receipt can be sealed, block publication of
  qualifying cohort results and all further submissions;
- verifies permissions, raw-to-public traceability, and retrieval from the
  protected locator;
- keeps publication eligibility read-only, then performs two fresh live
  eligibility passes before creating an eligible decision anchor; and
- pins staging-directory identity across atomic no-replace publication, verifies
  after rename, and rolls the public destination back to absence if post-commit
  verification or parent-directory durability fails.

Test success, nonzero exit, timeout, cancellation, partial output, missing
telemetry fields, telemetry death, malformed records, symlinks, path escape,
secrets, interrupted sealing, duplicate IDs, and retrieval mismatch with fake
or short commands before any model download. Test watchdog loss explicitly: it
must request owned cancellation, stop further submissions when ownership cannot
be proven, and retain the incomplete attempt or its capture-failure receipt
rather than reporting a pass.

The [Phase 2A tooling contract](cuda-campaign-phase2-tooling.md) records the
implemented interfaces and closed adversarial-review findings. Implementing
the Phase 4 source authority does not mean campaign Phase 4 ran: no production
Phase 4 artifact was collected. Development results are not an Ubuntu run,
CUDA measurement, or authorization to mutate the intended host.

### Phase 2B — sanitized recovery supplement

Phase 2B pinned merged source
`f6a58612263ccd1b7284ffa9f5460631ba64c2e1` and used only its reviewed
sanitizer, custody, eligibility, and publication paths against protected Phase
0 copies. The [dated supplement](evidence/2026-08-09-cuda-phase0-recovery-supplement/README.md)
gives all 40 expected rows a disposition, preserves the two known `not-found`
limitations, and binds the exact sealed artifact, two verified copies, full
retrieval, retention, durable independent review, finalization, and eligible
publication decision. The original August 6 packet remains immutable.

No Linux connection, Ubuntu-host mutation, model download, GPU workload, or
new empirical run occurred. Phase 2B supplies a sanitized recovery supplement,
not a new target-runtime result. Phase 3 subsequently completed in source;
Phase 4 is the next required gate.

### Phase 3 — explicit method selection and measurement contracts

**Complete in `aptus.training-plan.v6`.** Domain, API, CLI, and workbench
selection now choose one complete viable candidate, create a new plan identity,
and fail closed on stale, rejected, nonselectable, already-selected, or mutated
identities while preserving policy, inspection, and evidence bindings. Planning
and compiled CUDA configuration bind the optimizer-step target, independent
split/training/data-order seeds, and optional explicit micro-batch and
accumulation controls. Full CUDA training uses the compiled optimizer-step
target, rejects overrides, writes checkpoint control bindings, and emits
separate training/evaluation micro-iteration, completed-step, example, padded,
non-padding, and supervised-token counters plus monotonic per-step progress.
Static validation and parent completion verification enforce those bindings.
The Phase 2A watchdog remains only an emergency ceiling. This completion is
source-contract evidence, not a target-host result.

Use only the Phase 3 selection interface; generated plan or bundle files must
never be hand-edited. Every qualifying slot must bind its explicit optimizer
target, split/training/data-order seeds, micro-batch and accumulation values,
exact counters, and per-step progress record through the selected plan and
compiled bundle. Add any other proposed sweep axis to the planner before the
sweep rather than mutating generated trainer configuration out of band. The
Phase 2A safety watchdog remains an emergency ceiling and cancellation
mechanism, not a training-duration controller.

### Phase 4 — nonqualifying rehearsal and freeze

1. Run the capture harness against a fake command, then the shortest CUDA
   synthetic preflight.
2. Verify stream completeness, timestamps, telemetry coverage and overhead,
   sealing, sanitization, checksum verification, backup, and raw retrieval.
3. Freeze one clean source commit and tree after Phases 2 and 3 merge.
4. Record the exact host and driver facts and create clean, separately bound
   dependency environments for unquantized and bitsandbytes paths when their
   closures differ. Capture provisioning as separate setup evidence when its
   time will be published.
5. On that Ubuntu host and frozen source, run every applicable repository gate
   through the capture harness. Retain exact commands, Python/Node/tool versions,
   exit codes, pass/fail/skip counts, durations, byte-exact transcript digests
   and sizes, raw-manifest bindings, protected locators, retention-policy and
   receipt bindings, two-copy verification, and verified retrieval. Mark the
   macOS-only desktop build not applicable with its reason; do not treat a
   development-Mac gate run as the Ubuntu transcript.
6. Resolve and inspect every exact model/tokenizer revision before downloading
   campaign weights. Freeze the deterministic datasets and split assignments.
7. Generate the complete attempt ledger, exact cell counts, promotion rules,
   paired seeds, and randomized complete blocks before seeing
   measured outcomes.

Any capture, schema, source, environment, fixture, or stop-rule change after
this point invalidates the rehearsal and requires a new freeze.

### Phase 5 — repeatability anchor

**Complete with the anchor established.** A successful replacement cohort ran
all five predeclared measured slots in frozen order with no replacements. Every
slot completed 128 non-skipped optimizer steps with native outcome `passed`,
`protocol-valid` evidence, healthy telemetry, a verified seal, an
off-experiment-host copy, and a verified fresh retrieval. Duration MAD/median,
duration maximum/minimum, peak-device-memory range, telemetry coverage, and
maximum telemetry gap all passed the Phase 1 common stability contract. The
[successful sanitized packet](evidence/2026-08-10-cuda-phase5-repeatability-anchor/README.md)
therefore establishes the exact-host anchor and authorizes Phase 6 for the
exact frozen scope.

The earlier [stopped cohort](evidence/2026-08-09-cuda-phase5-repeatability-anchor/README.md)
remains immutable failure history. Its conditioning capture failed closed and
its five measured slots did not start. It is not included in the successful
cohort aggregate and was not overwritten or silently discarded.

Use one exact SmolLM2 135M revision, the frozen synthetic benchmark dataset,
LoRA, BF16 or the planner-selected supported precision, and `single` placement.
An optional warm-up is labeled and excluded. Then run exactly five predeclared
measured attempts with distinct state and output paths. Do not replace any
non-passing, guard-blocked, unknown, or capture-invalid started slot, or any
planned-not-started slot. Apply the batch pass threshold and stopping rule
frozen in Phase 1; if it is not met, report that the repeatability anchor was
not established and diagnose before scheduling a new cohort.

The August 6 acceptance is historical input and is not counted among these five
repetitions because it predates this capture protocol. Publish every slot with
its separate slot, native-outcome, and evidence statuses. Advance only after the
exactly five-attempt batch, its predeclared decision, and its raw retrieval proof
receive independent review.

### Phase 6 — same-model method matrix

**Complete with no promoted method.** The
[sanitized Phase 6 packet](evidence/2026-08-10-cuda-phase6-method-matrix/README.md)
retains all 32 predeclared slots with no replacements. Full fine-tuning was
blocked after its conditioning evidence was capture-invalid; Int8 LoRA and
QLoRA were not admitted on the exact host; and LoRA produced one qualifying
pass, one safety cancellation for unrelated GPU activity, and one activated
but unlaunched slot whose execution configuration did not match the initial
source freeze. No method met the frozen three-of-three promotion rule, so no
confirmatory slot started and Phase 7 is not authorized.

Hold model revision, tokenizer, data and split, loss masking, sequence length,
effective batch, Phase 3 optimizer-step target, seed policy, checkpoint rule,
capture, host, and idle/cooldown protocol constant.

| Cell | Required path | Initial expectation on this host |
| --- | --- | --- |
| `full` | BF16, `single` | Run only if the exact candidate and live admission pass |
| `lora` | Supported precision, `single` | Repeatability anchor |
| `int8-lora` | bitsandbytes 8-bit base, `single` | Run only after exact kernel and environment validation |
| `qlora` | bitsandbytes 4-bit base, `single` | Run only after exact kernel and environment validation |

Run exactly three predeclared exploratory attempt slots per admitted cell; do
not replace a non-passing or missing attempt. Freeze and disclose every
method-specific environment, learning-rate, adapter-rank/alpha, optimizer,
parameter-scope, and micro-batch difference as a comparison limitation. Apply
the Phase 1 promotion/stability rule without inspecting a post-hoc subset. For
cells that satisfy it, schedule exactly five randomized complete measured
blocks: each block contains one attempt slot for every promoted method in a
frozen randomized order and uses the paired seed from the predeclared schedule.
Preserve every non-passing, capture-invalid, or planned-not-started slot and
never replace it. This prevents method order from tracking time or temperature.
Exploratory results remain published but are excluded from confirmatory
aggregates. Resource results are not a quality ranking.

### Phase 7 — scale staircase and architecture breadth

Start with the same Llama-family SmolLM2 revisions at approximately 135M, 360M,
and 1.7B parameters. For each size, begin with the stable LoRA configuration,
then attempt only the Phase 6 methods that have valid candidates. At every new
cell, proceed through static validation, dependency, model-data,
measured-preflight, and pilot before confirmed training. For each cell admitted
to characterization, schedule exactly three exploratory attempt slots and do
not replace a non-passing slot. Do not call that batch confirmatory
repeatability.

If 1.7B passes, continue through a Phase 4-predeclared sequence of progressively
larger, provider-inspected dense Llama-family artifacts until the first safe
admission refusal or bounded-pilot capacity failure, or until the explicit
maximum model/download/storage budget is reached. Exact artifacts and revisions
must be approved before the measured cohort; do not choose the next model after
seeing a favorable result.

After the same-family size ladder is reviewed, select at most one small,
provider-inspected dense artifact from each additional currently recognized
family such as Mistral, Gemma, or dense Qwen3. Under the current policy
snapshot, the reviewed Qwen2 policy permits only its exact MLX-LM path and does
not yield a CUDA candidate; exclude Qwen2 unless a later reviewed snapshot and
inspection explicitly produce a viable CUDA candidate. Do not
schedule any artifact because its name appears compatible: the exact
inspection, policy, candidate, revision, license, and target modules must pass
first. MoE and multimodal artifacts remain outside this campaign.

Stop a model/method staircase at the first safe admission refusal or
bounded-pilot runtime capacity failure. Record that point; do not force the next
larger artifact or launch a full run merely to obtain a failure.

### Phase 8 — guarded configuration frontier

Choose a stable model/method cell with thermal and memory headroom. Change one
planner-controlled axis at a time:

1. sequence length;
2. effective batch; and
3. Phase 3 planner-controlled micro-batch or accumulation while preserving and
   reporting effective batch.

Use a predeclared increasing ladder bounded by the model context and Aptus
admission. Recompile for every point and never lower the normal 2 GiB CUDA
reserve to extend the ladder. Prefer a measured admission refusal as the upper
endpoint. If runtime characterization is necessary, run only the bounded pilot
at the next point; never launch confirmed full training to seek an OOM. An OOM
is an unplanned censored failure, not a campaign target. The endpoint does not
prove that every intermediate or larger configuration fails. Adapter rank,
optimizer, packing, and other absent planner axes require their own
implementation and review before they can be swept.

### Phase 9 — endurance and job control

Select a point below the frontier that meets the Phase 1-predeclared numeric
headroom margins above the hard free-VRAM, available-RAM, free-disk, and
temperature stops across every qualifying input run. It must also have no
thermal throttling, Xid error, applicable hardware error, or telemetry gap.
Record the exact observed margins and selection calculation, then schedule
exactly three measured endurance attempt slots without replacement. Predeclare
the several-hundred-update rule through the Phase 3 optimizer-step contract;
observed wall time does not become an enforced-duration claim. The safety
watchdog remains only an emergency ceiling. Monitor for drift in step time,
memory, temperature, power, clock state, loss, and artifact growth through the
Phase 3 per-step monotonic progress record.

Separately exercise managed cancellation, same-user global lease exclusion,
stale-owner recovery, parent verification, and crash-safe pending-completion
promotion on Ubuntu with controlled test jobs. Do not sabotage a qualifying
training result to create recovery evidence. Semantic CUDA export loading or
generation remains an open gate until its own contract is implemented and
tested; structural export alone must stay labeled structural.

### Phase 10 — aggregation, review, and publication

For each comparison cell, publish the complete attempt count,
execution-configuration identities, and individual values. For measured
blocks, apply
exactly the median, minimum, maximum,
uncertainty, missing-result, failure, outlier, and quantile rules frozen in
Phase 1; do not choose a treatment after seeing results or hide outliers.
Telemetry p95 describes samples within runs, not confidence across five
repetitions.

Validate the complete slot ledger before aggregation: slot IDs are unique and
every frozen slot appears exactly once; `Planned = Started +
Planned-not-started`; `Started` equals the sum of all native execution outcomes;
and, independently, `Started = Protocol-valid + Capture-invalid`. Only the
intersection of native `passed` and evidence `protocol-valid` contributes a
qualifying pass.

Publish separate conclusions for:

- observed workflow completion rate and gate outcomes within the exact batch;
- exact-host time and resource measurements;
- method comparisons under their disclosed controls and confounds;
- model and configuration fit boundaries;
- endurance and job-control behavior; and
- failures, refusals, missing fields, and untested rows.

An independent reviewer must verify raw retrieval, hashes, sanitization,
calculations, claim wording, and the mapping to release gates. Only after the
dated packet merges may current-capability, capability-matrix, claim-language,
changelog, or release-gate result text broaden. A later multi-GPU campaign is
still required for DDP and conditional LoRA FSDP.

## Immediate next actions

1. Preserve the frozen Phase 1 protocol and its machine-readable companion as
   design authority. They authorize no runtime execution.
2. Preserve the completed [Phase 2 tooling and recovery-publication
   record](cuda-campaign-phase2-tooling.md) and dated [sanitized
   supplement](evidence/2026-08-09-cuda-phase0-recovery-supplement/README.md).
   They record no new Ubuntu result.
3. Preserve the completed Phase 3 selection and measurement-control contracts.
4. Preserve the completed Phase 4 rehearsal, repository-gate evidence, source
   freeze, host/environment bindings, fixtures, and frozen run order.
5. Preserve the independently reviewed successful Phase 5 packet and its
   separate immutable stopped-cohort history.
6. Execute Phases 6 through 9 in order, sealing and reviewing each batch before
   expanding the matrix.
7. Publish Phase 10 packets and update claims only to the exact evidence boundary.

## Related documentation

- [Operations index](index.md)
- [Release gates](release-gates.md)
- [Release evidence template](release-evidence-template.md)
- [Operator checklist](operator-checklist.md)
- [State, storage, and retention](state-storage-retention.md)
- [Phase 2A tooling contract](cuda-campaign-phase2-tooling.md)
- [CUDA campaign protocol](../reference/cuda-campaign-protocol.md)
- [CUDA campaign protocol machine companion](../reference/cuda-campaign-protocol.v1.json)
- [Preflight and calibration](../methodology/preflight-calibration.md)
- [Design an evaluation](../guides/design-an-evaluation.md)
- [Method registry](../reference/method-registry.md)
- [Model-policy snapshot](../reference/model-policy-snapshot.md)
- [Capability matrix](../reference/capability-matrix.md)
- [2026-08-06 SmolLM2 CUDA LoRA single-device acceptance](evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/README.md)
