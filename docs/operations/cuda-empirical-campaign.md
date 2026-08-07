# RTX 3050 CUDA Empirical Evidence Campaign

> **Status:** Active experiment plan | **Authority:** Canonical operational plan for bounded CUDA evidence; non-normative for current capability | **Applies to:** Aptus 0.2 single-device CUDA characterization on the intended Ubuntu RTX 3050 host | **Audience:** Operators, maintainers, and evidence reviewers | **Owner:** CUDA runtime and release evidence | **Last reviewed:** 2026-08-07 | **Review by:** Before the first qualifying run, after any capture or selection contract changes, or by 2026-09-07

This is the one execution plan for the next CUDA evidence campaign. It combines
the remaining roadmap work, release gates, evidence-packet requirements,
storage rules, operator procedure, calibration methodology, and fair-comparison
guidance into one ordered program without taking authority away from those
documents.

The plan schedules work; it does not assert that a scheduled run passed. Only a
reviewed, dated, checksum-covered record under [`evidence/`](evidence/) can
establish a measured result. Current capability language remains unchanged
until such a record passes independent review.

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

1. Never edit a generated plan, `recommended` candidate, bundle configuration,
   or evidence file to force a matrix cell. Aptus currently compiles the
   recommended candidate; Phase 3 must add an explicit, validated selection
   contract before comparing alternate viable methods.
2. Never reuse a run output directory or overwrite a prior state root. Every
   repetition receives fresh state, bundle, run, and capture locations.
3. Never install, update, clean, or start Aptus on the Ubuntu host until Phase 0
   has produced two checksum-verified recovery copies of the August 6 raw
   records in separate failure domains, including one off-host, Phase 2 has
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
7. Never report tokens per second until exact padded and supervised token
   counters are implemented and bound. Until then, report the runtime's exact
   step or example rates and the limitation.

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
mutation. Phases 2 and 3 are separate reviewable code changes, and both must
merge before a qualifying GPU run.

| Phase | Outcome | May start when | Exit evidence |
| ---:| --- | --- | --- |
| 0 | Recover and protect prior Ubuntu raw evidence without publishing it yet | This plan is approved | Complete private inventory, digest comparison, two verified copies, and off-host retrieval |
| 1 | Freeze protocol, record fields, retention, fixtures, and claim boundary | Phase 0 recovery may proceed in parallel, but no host mutation occurs | Reviewed plan, recovery-supplement schema, and sanitizer decisions |
| 2 | Implement and test capture tooling, then publish the recovered-evidence supplement | Phase 1 fields are frozen and Phase 0 inventory is complete | Full repository gates, fake-command evidence, and independently reviewed recovery supplement |
| 3 | Implement explicit candidate selection and exact measurement contracts | Phase 1 selection semantics are approved | Full gates and candidate-identity mutation tests |
| 4 | Rehearse the harness and freeze the campaign source, host, environments, models, data, and run order | Phases 0, 2, and 3 pass | Nonqualifying rehearsal, captured Ubuntu repository gates, and successful retrieval |
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
8. Preserve the private recovery inventory and its digest as Phase 2 input. Do
   not publish it with an improvised schema or sanitizer before Phase 1 review.

Phase 0 is complete only when every existing digest has a disposition, two
verified copies exist in separate failure domains, the off-host copy has passed
retrieval, and the host can remain unchanged without risking the only raw copy.
Phase 0 completion alone does not authorize host mutation.

### Phase 1 — protocol and schema decisions

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

### Phase 2 — evidence-capture infrastructure

Add an opt-in experiment harness under `tools/`; do not change ordinary Aptus
execution semantics merely to satisfy the campaign. The implementation must:

- launch an exact command while capturing complete output and timing;
- emit one monotonic event ledger with UTC mapping for command, managed-job,
  pilot-phase, training, export, verification, telemetry, sealing, and cooldown
  boundaries, then derive segmented summaries from those markers;
- capture each environment-provisioning or model-download command as its own
  setup record when setup timing will be published;
- copy and seal terminal Aptus job records, complete logs, reports, metrics,
  manifests, and selected artifacts;
- run a low-overhead NVIDIA and host telemetry sidecar;
- enforce every live safety trigger and the frozen deadline through the owning
  `JobService`: record detection, cancellation-request, confirmed process-group
  termination, and lease-reconciliation timestamps by exact job ID, then seal
  the exact terminal disposition; never signal a stale numeric PID directly,
  and block further submissions when ownership or termination is uncertain;
- create no-clobber raw manifests, sanitization projections, aggregates, and
  `SHA256SUMS`;
- when normal capture or sealing cannot complete, write an immutable
  capture-failure receipt through a protected fallback path with the
  attempt-slot and run IDs, stable failure code, available-file inventory,
  missing-field list, SHA-256, byte size, and any recoverable locator; if neither
  a canonical raw manifest nor that receipt can be sealed, block publication of
  qualifying cohort results and all further submissions;
- verify permissions, raw-to-public traceability, and retrieval from the
  protected locator; and
- fail the qualifying-run decision when capture, sealing, sanitization, or
  retrieval is incomplete.

After those tools and their allowlist pass review, create a new dated sanitized
recovery-supplement packet from the Phase 0 inventory. Give every expected item
its disposition, recovered byte size when available, manifest binding, opaque
vault locator, retention-policy and receipt bindings, copy-verification result,
and retrieval date/result. Keep the original August 6 packet immutable. The
supplement must pass independent sanitization and raw-traceability review before
any Ubuntu-host mutation.

Test success, nonzero exit, timeout, cancellation, partial output, missing
telemetry fields, telemetry death, malformed records, symlinks, path escape,
secrets, interrupted sealing, duplicate IDs, and retrieval mismatch with fake
or short commands before any model download. Test watchdog loss explicitly: it
must request owned cancellation, stop further submissions when ownership cannot
be proven, and retain the incomplete attempt or its capture-failure receipt
rather than reporting a pass.

### Phase 3 — explicit method selection and measurement contracts

A fair method matrix is impossible while compilation can select only
`plan.recommended` and a method preference merely influences ranking. Add an
explicit API/domain/CLI or equivalent contract that selects one complete viable
candidate and produces a new plan identity. It must reject a stale, rejected,
nonselectable, or mutated candidate and preserve the policy and evidence chain.
Generated plan or bundle files must never be hand-edited.

Add exact padded and supervised token counters before publishing token
throughput. If that work is deferred, the campaign remains valid but publishes
only the exact existing step/example rates and wall times. Add any planner axes
needed for a proposed sweep before the sweep; do not mutate generated trainer
configuration out of band. In particular, current planning derives micro-batch
and accumulation rather than accepting them as independent inputs, exposes no
optimizer-step target, rejects enforced wall time, and forbids a full-run
`--max-steps` override. Phase 8 needs reviewed micro-batch/accumulation controls
before sweeping them. Phase 9 needs a reviewed optimizer-step or graceful
deadline contract before claiming an enforced endurance duration. The Phase 2
safety watchdog is an emergency ceiling and cancellation mechanism, not a
training-duration controller.

Current CUDA bundles also compile seed `17`, and full training rejects the
pilot-only `--seed` override. Add a reviewed seed input through the planning,
bundle, runtime, and validation contracts before any multi-seed paired schedule;
otherwise every slot must declare the fixed seed and no conclusion may imply
seed breadth. Add fixed-window or per-step progress timestamps and rates before
Phase 9 claims step-time drift; aggregate trainer rates alone cannot establish
drift.

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

Hold model revision, tokenizer, data and split, loss masking, sequence length,
effective batch, compiled epochs, derived optimizer-step count, seed policy,
checkpoint rule, capture, host, and idle/cooldown protocol constant. Hold an
explicit optimizer-step or supervised-token target only after Phase 3 adds and
validates that contract.

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
3. only after Phase 3 adds explicit planner inputs, micro-batch or accumulation
   while preserving and reporting effective batch.

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
exactly three measured endurance attempt slots without replacement. If Phase 3
has added a reviewed optimizer-step or graceful-deadline contract, predeclare a
30-to-60-minute or several-hundred-update rule through that contract. If it has
not, freeze dataset size and compiled epochs and report elapsed time as observed;
do not claim Aptus enforced a duration or update target. The safety watchdog
remains only an emergency ceiling. Monitor for drift in step time, memory,
temperature, power, clock state, loss, and artifact growth only through the
fixed-window or per-step progress contract added in Phase 3; otherwise report
aggregate rates without a drift conclusion.

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

1. Merge the documentation change that establishes this plan and its cross-links.
2. Boot the Ubuntu host only for read-only Phase 0 recovery before any repository
   pull, install, cleanup, or Aptus execution.
3. Freeze the Phase 1 identities, schemas, sanitizer, retention receipts,
   thresholds, counts, schedules, and decision rules.
4. Implement and review Phase 2 capture infrastructure, then use its reviewed
   sanitizer to publish and independently verify the Phase 0 recovery
   supplement. Keep the Ubuntu host otherwise unchanged until that merges.
5. Implement and review the Phase 3 candidate-selection, seed, progress, and
   measurement contracts.
6. Merge the Phase 2 and 3 code changes, run the complete repository gates with
   retained transcripts, and perform the Phase 4 nonqualifying rehearsal.
7. Freeze the exact source, host, environments, datasets, models, and run order.
8. Execute Phases 5 through 9 in order, sealing and reviewing each batch before
   expanding the matrix.
9. Publish Phase 10 packets and update claims only to the exact evidence boundary.

## Related documentation

- [Operations index](index.md)
- [Release gates](release-gates.md)
- [Release evidence template](release-evidence-template.md)
- [Operator checklist](operator-checklist.md)
- [State, storage, and retention](state-storage-retention.md)
- [Preflight and calibration](../methodology/preflight-calibration.md)
- [Design an evaluation](../guides/design-an-evaluation.md)
- [Method registry](../reference/method-registry.md)
- [Model-policy snapshot](../reference/model-policy-snapshot.md)
- [Capability matrix](../reference/capability-matrix.md)
- [2026-08-06 SmolLM2 CUDA LoRA single-device acceptance](evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/README.md)
