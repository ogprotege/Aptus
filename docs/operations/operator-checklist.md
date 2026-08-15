# Operator Checklist

> **Status:** Active | **Audience:** Local CUDA and Apple Silicon operators | **Authority:** Operational | **Applies to:** Aptus 0.2 | **Owner:** Runtime operations | **Last reviewed:** 2026-08-07 | **Review by:** 2026-10-27

Use this checklist for one Aptus bundle on one trusted-user host. Aptus is not a
remote scheduler or multi-user service. The checklist does not replace the
exact bundle runbook or release gates.

For repeated RTX 3050 characterization, follow the [canonical CUDA empirical
campaign](cuda-empirical-campaign.md) in addition to this per-bundle checklist.
That campaign owns run ordering, repetitions, telemetry, stop rules, protected
raw capture, and aggregation. Do not improvise those fields between runs.

## Current operating boundary

- CUDA supports guarded full, LoRA, eight-bit LoRA, and QLoRA paths on an
  external CUDA host.
- Apple Silicon supports conditional single-device MLX-LM LoRA and QLoRA
  through uninterrupted pilot and full-duration adapter training.
- An MLX `pilot-pass` proves a two-update uninterrupted run plus fresh-process
  adapter reload and bounded generation. It does not prove crash resume.
- PyTorch MPS has no compiler. LM Studio and oMLX are inference-only.
- Standalone `aptus serve` generates a new session token for every launch and
  prints the workbench origin without that token. It protects the API with a
  cookie or bearer header, but adds no tenant isolation or remote-user policy.
- The host-global lease coordinates Aptus processes for one local user. It does
  not reserve accelerator or unified-memory capacity against unrelated programs.
- Package-free bundle entrypoints validate their embedded frozen model-policy
  snapshot. That proves integrity and saved-decision parity, not current host
  policy. Installed Aptus separately enforces current registry currency.
- Full training requires a current passing pilot and explicit confirmation.
- Full-training resume is unsupported.
- Structural export verification does not establish model quality or safety.
- A Mac does not provide CUDA evidence. Use the MLX path locally or hand a CUDA
  bundle to its external target host.

Keep the service on `127.0.0.1`. `--allow-non-loopback` preserves session
authentication but sends its credential over plain HTTP. It does not add TLS,
tenant isolation, filesystem scoping, or worker isolation.

## Before starting the local service

Run the read-only environment check first:

```bash
aptus doctor --state-dir .aptus-state
```

Exit `0` means a compatible training interpreter was observed. Exit `2` means
action is required. The doctor installs nothing.

- [ ] Bind `aptus serve` to `127.0.0.1`, `localhost`, or `::1`.
- [ ] Protect the printed bearer token as a credential. The printed workbench
      URL must not include `aptus_session_token`.
- [ ] If you opt into the query handoff, open that URL and verify the first
      response redirects to the same path without `aptus_session_token`.
- [ ] Verify a successful handoff sets an HttpOnly, SameSite Strict session
      cookie. Prefer `Authorization: Bearer TOKEN` for API clients.
- [ ] Verify health and static assets remain public, while an unauthenticated
      product API request returns `403 desktop_session_required`.
- [ ] For automation, send `Authorization: Bearer TOKEN` and never place the
      token in job JSON, plan JSON, or source data.
- [ ] Confirm Uvicorn access logging is disabled by the CLI. Still protect
      terminal capture, browser history, and process-supervisor output.

The Mac app uses a different handoff. Its exact-origin native host installs the
cookie before WebKit makes its first request, so the desktop token never enters
a URL.

If non-loopback serving is unavoidable, place it behind approved TLS and
network controls. A network observer can steal a cookie or bearer token from
the built-in plain-HTTP server.

## Before receiving a bundle

- [ ] Confirm the operator, requestor, and intended use.
- [ ] Confirm model and dataset rights.
- [ ] Confirm the model is pinned to an immutable 40 to 64 character hexadecimal
      commit identifier.
- [ ] Confirm the dataset digest and sensitivity classification.
- [ ] Confirm the training runtime, target host, accelerator, memory model,
      disk, and required reserve.
- [ ] Confirm the selected method, precision, quantization, placement, world
      size, and effective batch.
- [ ] Confirm the v5 plan records its compatibility decision and lowercase
      `model_policy_snapshot_sha256`; do not accept a hand-edited plan or
      snapshot.
- [ ] Confirm the output and retention budget for caches, pilot artifacts,
      CUDA checkpoints, MLX weight snapshots, logs, ZIP archives, and final
      exports.
- [ ] Confirm that no credential appears in plan JSON, data, generated source,
      or a job request.

## Before installing dependencies

- [ ] Place the bundle and archive in access-controlled storage.
- [ ] Verify `bundle-manifest.json` with static validation.
- [ ] Verify `policy/model-policy-snapshot.v1.json` exists, is canonical
      `aptus.model-policy-snapshot.v1`, and is manifested at that exact path.
- [ ] Confirm the snapshot bytes, plan `model_policy_snapshot_sha256`, manifest
      `policy_snapshot_sha256`, and manifest file entry use the same lowercase
      SHA-256 digest. Installed-host validation separately compares the current
      registry digest.
- [ ] Review `decision-report.md`, `requirements.txt`, generated programs, and
      configuration.
- [ ] Use a new isolated Python environment outside the bundle.
- [ ] For managed jobs, install Aptus and the generated requirements in that
      same environment. Confirm `aptus` resolves from its interpreter.
- [ ] Verify the exact runtime Python, package-index, and model-cache access.
- [ ] For CUDA, verify the driver and CUDA runtime. For MLX-LM, verify Apple
      silicon macOS plus the pinned MLX and MLX-LM versions.
- [ ] Verify free disk includes model downloads, environment files, pilot
      workspaces, CUDA checkpoints or MLX weight snapshots, and final export.
- [ ] Stop or account for unrelated accelerator and memory-intensive jobs. They
      do not honor the Aptus lease.

Generated `requirements.txt` contains exact direct pins. Retain the resolved
installed-environment binding because transitive distributions are selected by
the installer.

Treat `POLICY_SNAPSHOT_MISSING`, `POLICY_SNAPSHOT_JSON_ERROR`,
`POLICY_SNAPSHOT_CONTRACT`, `POLICY_SNAPSHOT_NONCANONICAL`,
`POLICY_SNAPSHOT_DIGEST`, and `POLICY_SNAPSHOT_PATH` as fail-closed findings.
Snapshot, plan, or manifest corruption requires a trusted recompile. A valid
host-only digest difference means the coherent v5 plan is no longer current and
requires a new plan and bundle; do not edit the old digest chain.

## Know the state locations

The CLI and service default to `.aptus-state` in the process working directory.
With the default root:

```text
.aptus-state/
  plans/                 persisted API plans
  current-bundle.json    restorable bundle pointer
  current-project.json   current named-project pointer
  runtime-config.json    validated external interpreter choices
  projects/              manifests and immutable revision records
  jobs/
    job_<id>.json        atomic managed job record
    job_<id>.log         complete managed job log
  quarantine/            contained corrupt records and reason receipts
```

Project recovery creates a new revision and never restores training
authorization. Revalidate and reconfirm before a later train action.

The bundle holds its own mutable validation report, pilot output, and run
directories. The per-user lease directory is
`$XDG_RUNTIME_DIR/aptus/aptus-gpu-lease-<hash>/` when `XDG_RUNTIME_DIR` is a
secure user runtime directory, otherwise `~/.aptus/run/aptus-gpu-lease-<hash>/`.
It is not created under world-writable `/tmp`.
Do not delete a lease or edit a job record to clear an apparent conflict.
Inspect the recorded owner and let Aptus reconcile stale state.

Use `--state-dir` consistently when you choose a nondefault root. The
host-global lease still prevents another Aptus state root for the same user from
starting a competing accelerator action.

## Run the ordered actions

| Order | Managed command | Required successful state | Inspect before continuing |
|---:|---|---|---|
| 1 | `aptus run BUNDLE --action dependency` | `dependency-pass` | Direct pins, Python/platform binding, installed distribution closure |
| 2 | `aptus run BUNDLE --action model-data` | `model-data-pass` | Pinned model structure, tokenizer, target modules, every canonical row, trainable scope; for MLX-LM, packed-checkpoint admission and `model-data-evidence.json` |
| 3 | `aptus run BUNDLE --action preflight` | `measured-preflight-pass` | CUDA synthetic evidence or MLX bounded real-input smoke, runtime memory, and bindings |
| 4 | `aptus run BUNDLE --action pilot` | `pilot-pass` | CUDA: two fresh phases and checkpoint continuation. MLX: uninterrupted two-update run and fresh-process adapter generation |
| 5 | `aptus run BUNDLE --action train --confirm-full-train` | `measured-run-pass` after parent verification | Current admission, runtime-specific full metrics, final export, and immutable evidence |

Replace `BUNDLE` with the resolved bundle directory. Wait for each command to
reach a terminal state. Between actions, inspect:

```bash
aptus jobs
aptus jobs --id JOB_ID
```

The log path in the job record is authoritative. Preserve the complete log,
not only the returned tail.

A package-free `validate.py` pass does not waive installed-host policy checks.
Managed submission, pilot authorization, worker launch, and the completion
verification and promotion transaction recheck current registry currency. An
HTTP `409 replan_required` from saved-plan load, compile, project recovery, or
managed job submission requires replanning and recompilation.

For an MLX-LM bundle, continue only when each earlier action passes. Its pilot
and full run both start from the pinned base and run without interruption. Do
not supply a resume argument.

## Dependency action

- [ ] Exact direct package versions are installed.
- [ ] The environment binding contains Python, platform, direct constraints,
      and runtime distribution closure.
- [ ] No unreviewed package substitution occurred.
- [ ] Runtime imports and exact versions pass. CUDA driver compatibility is
      present for CUDA; MLX Metal capability is present for MLX-LM.

## Model-data action

- [ ] The exact model and tokenizer resolve at the pinned revision.
- [ ] Loaded parameter count and structural facts match within the documented
      contract.
- [ ] Every required adapter target module exists.
- [ ] The prepared method has a positive, finite, correctly scoped trainable
      set.
- [ ] Every canonical row transforms with non-empty supervision.
- [ ] Credentials remained in the underlying model stack, not the bundle.
- [ ] For `mlx-lm`, live unified-memory admission passed before the model
      loaded. The completed action wrote mutable runtime artifact
      `model-data-evidence.json` under `aptus.mlx-model-data-evidence.v1`,
      containing an `aptus.mlx-unified-memory-admission.v2` record. The
      validation report binds the artifact's current SHA-256.

Model-data validation does not enter training mode and does not prove
accelerator fit. MLX QLoRA must obtain four-bit eligibility from the pinned
model metadata, not a CUDA-style device flag.

On `mlx-lm` this action measures the packed safetensors shards and compares live
available unified memory against the packed-checkpoint-adjusted candidate
estimate plus `max(plan reserve, 8 GiB)`. If the shortfall is positive it refuses
before any weight load and reports exact required, available, and shortfall byte
counts. Treat that refusal as a legitimate fail-closed outcome and record it: the
2026-07-28 Qwen3 30B-A3B attempt stopped here.

## Measured preflight action

- [ ] The method, precision, quantization, distribution, world size, and device
      indices match the selected candidate.
- [ ] The runtime-specific measured peak is positive.
- [ ] CUDA census evidence, or the MLX exact target binding, adapter delta,
      adapter manifest, and runtime bindings, match the selected candidate.
- [ ] Current unrelated accelerator and unified-memory pressure is understood.

CUDA preflight uses a synthetic method path. MLX preflight uses the exact pinned
model and compiled MLX data for at most eight iterations, but its scope remains
`bounded-compiler-smoke-not-pilot-evidence`.

## Pilot action

For CUDA:

- [ ] Both phases ran in fresh processes.
- [ ] Both phases bind the same plan, candidate, environment, hardware, and
      trainable census.
- [ ] Losses are finite and steps are positive.
- [ ] Phase one checkpoint evidence passes.
- [ ] Phase two proves the expected step continuation.
- [ ] Pilot artifact and export manifests pass.
- [ ] Measured CUDA peaks and checkpoint/export sizes are recorded.

For MLX-LM:

- [ ] The pilot ran uninterrupted from the pinned base against the exact bound
      train and validation files.
- [ ] At least two optimizer updates completed, and train and validation losses
      are finite.
- [ ] Every planned target in every layer has exactly one LoRA A/B pair, with no
      other trainable tensor.
- [ ] MLX peak memory and adapter delta are positive.
- [ ] Live unified-memory admission passed with the required reserve.
- [ ] The fresh action-owned marker, metrics, adapter pair, reload evidence, and
      artifact manifest bind exact paths, sizes, and hashes.
- [ ] A fresh child loaded the pinned base plus adapter and generated one to four
      tokens with a positive MLX peak.
- [ ] Evidence says `execution_semantics: uninterrupted` and
      `resume_supported: false`.

Do not authorize training after any pilot failure. Preserve the failed metrics,
logs, and artifacts for diagnosis.

## Train admission

Immediately before submission:

- [ ] Confirm the expensive full-run action with the requestor.
- [ ] Confirm the current bundle manifest and plan identity.
- [ ] Confirm the embedded snapshot still matches the plan and manifest, and the
      installed host accepts its decision and digest under the current
      model-policy registry.
- [ ] Confirm current CUDA identities and free VRAM.
- [ ] For MLX, confirm current available unified memory exceeds the measured
      pilot peak plus reserve.
- [ ] Confirm current free host RAM where applicable and free disk.
- [ ] Confirm the installed environment still matches the pilot.
- [ ] Confirm CUDA checkpoint trees, or MLX owned pilot metrics, adapter, reload
      evidence, and artifact manifests, remain intact.
- [ ] Confirm the unique run output path does not exist.

The submission transaction performs these checks while holding the job locks
and host-global lease. Cached UI authorization and a historical portable pass
are informational. MLX admission also checks current free disk against the plan
and measured pilot artifacts. If policy is non-current, preserve the old
artifact and create a new plan and bundle.

## Monitor training and verification

- [ ] Watch state, phase, log growth, device memory, host memory, disk, and
      unrelated workload changes.
- [ ] Treat `queued`, `running`, and `cancelling` as active.
- [ ] Treat phase `verifying` as parent-owned work within `running`.
- [ ] Do not infer success from a zero child exit.
- [ ] Do not edit or move the active run directory.
- [ ] Do not start another Aptus accelerator action under a different state
      root.

The parent must verify runtime-specific metrics and immutable artifacts before
completion. CUDA verifies ranks, trainable census, split evidence, and the
structural export tree. MLX verifies completed updates, finite losses, exact
target binding, memory and adapter deltas, fresh adapter generation, and its
final adapter export.

## Cancellation

Use the workbench or the job-cancel API owned by the live service. POSIX
cancellation targets the recorded process group and can escalate after its grace
period. The job remains `cancelling` while termination is reconciled.

If another service owns the job, cancel through that service. If the owner died
but a child remains live, inspect the recorded process identity before taking an
operating-system action. Never kill a PID based only on an old numeric value.

Cancellation is refused while the parent commits verified completion evidence.

## After a successful run

- [ ] Job state is `completed`.
- [ ] Validation report is `measured-run-pass`.
- [ ] Completion attestation names the expected run, plan, and candidate.
- [ ] Artifact integrity is reported as verified.
- [ ] Full-run metrics include finite losses, positive steps, census, split,
      ranks, and measured peaks where the runtime defines those fields.
- [ ] `final-export.json` covers the expected structural file tree.
- [ ] The unique run path, reports, logs, environment record, and manifest are
      retained together.
- [ ] Any quality statement is withheld until a separate evaluation passes.

## After failure or interruption

1. Preserve the job JSON, full log, report, plan, embedded policy snapshot,
   manifest, and run directory.
2. Classify the cause as policy-snapshot integrity, current-policy currency,
   dependency, model/data, method scope, accelerator or
   unified-memory capacity, distribution, CUDA checkpoint, MLX weight snapshot,
   split, export,
   cancellation, or parent verification.
3. Correct the environment, facts, data, or source implementation.
4. Recompile to a new bundle when compiler-managed content changed.
5. Repeat every invalidated action.
6. Submit a new full run with a new run ID.

Do not resume an arbitrary full checkpoint or MLX weight snapshot. Aptus 0.2
does not bind the complete state needed for general full-run resume. Every MLX
resume argument fails closed.

## Retention and cleanup

Aptus 0.2 has no automatic retention or cleanup policy. Establish one before
large runs. Never remove evidence needed to interpret an active job or an
artifact still in use. Record any manual deletion by exact path and retention
rule. Preserve the plan, embedded policy snapshot, manifest, and their digest
bindings together when retaining historical evidence. Protect all deleted
material according to its sensitivity and recovery requirements.

## Related documentation

- [Compile, validate, and run](../guides/compile-validate-run.md)
- [Inspect results](../guides/inspect-results.md)
- [Recovery and the resume boundary](../guides/resume-recover.md)
- [Security policy](../../SECURITY.md)
- [Release gates](release-gates.md)
