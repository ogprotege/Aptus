# Operator Checklist

> **Status:** Active | **Audience:** Local CUDA operators | **Authority:** Operational | **Applies to:** Aptus 0.2 | **Owner:** Runtime operations | **Last reviewed:** 2026-07-22 | **Review by:** 2026-10-22

Use this checklist for one Aptus bundle on one trusted-user host. Aptus is not a
remote scheduler or multi-user service. The checklist does not replace the
exact bundle runbook or release gates.

## Current operating boundary

- CUDA is the only execution backend.
- The service has no authentication, tenant isolation, or remote-user policy.
- The host-global lease coordinates Aptus processes for one local user. It does
  not reserve GPUs against unrelated programs.
- Full training requires a current passing pilot and explicit confirmation.
- Full-training resume is unsupported.
- Structural export verification does not establish model quality or safety.
- No real CUDA pilot has been completed on the current development Mac.

Keep the service on `127.0.0.1`. `--allow-non-loopback` records an
acknowledgment but does not add authentication, TLS, origin policy, or
filesystem isolation.

## Before receiving a bundle

- [ ] Confirm the operator, requestor, and intended use.
- [ ] Confirm model and dataset rights.
- [ ] Confirm the model is pinned to an immutable 40 to 64 character hexadecimal
      commit identifier.
- [ ] Confirm the dataset digest and sensitivity classification.
- [ ] Confirm the target host, CUDA devices, host RAM, disk, and required
      reserve.
- [ ] Confirm the selected method, precision, quantization, placement, world
      size, and effective batch.
- [ ] Confirm the output and retention budget for caches, pilot artifacts,
      checkpoints, logs, ZIP archives, and final exports.
- [ ] Confirm that no credential appears in plan JSON, data, generated source,
      or a job request.

## Before installing dependencies

- [ ] Place the bundle and archive in access-controlled storage.
- [ ] Verify `bundle-manifest.json` with static validation.
- [ ] Review `decision-report.md`, `requirements.txt`, generated programs, and
      configuration.
- [ ] Use a new isolated Python environment outside the bundle.
- [ ] For managed jobs, install Aptus and the generated requirements in that
      same environment. Confirm `aptus` resolves from its interpreter.
- [ ] Verify Python, driver, CUDA, package-index, and model-cache access.
- [ ] Verify free disk includes model downloads, environment files, pilot
      workspaces, retained checkpoints, and final export.
- [ ] Stop or account for unrelated GPU jobs. They do not honor the Aptus lease.

Generated `requirements.txt` contains exact direct pins. Retain the resolved
installed-environment binding because transitive distributions are selected by
the installer.

## Know the state locations

The CLI and service default to `.aptus-state` in the process working directory.
With the default root:

```text
.aptus-state/
  plans/                 persisted API plans
  current-bundle.json    restorable bundle pointer
  jobs/
    job_<id>.json        atomic managed job record
    job_<id>.log         complete managed job log
```

The bundle holds its own mutable validation report, pilot output, and run
directories. On POSIX, a per-user lease directory is created under `/tmp`.
Do not delete a lease or edit a job record to clear an apparent conflict.
Inspect the recorded owner and let Aptus reconcile stale state.

Use `--state-dir` consistently when you choose a nondefault root. The
host-global lease still prevents another Aptus state root for the same user from
starting a competing accelerator action.

## Run the ordered actions

| Order | Managed command | Required successful state | Inspect before continuing |
|---:|---|---|---|
| 1 | `aptus run BUNDLE --action dependency` | `dependency-pass` | Direct pins, Python/platform binding, installed distribution closure |
| 2 | `aptus run BUNDLE --action model-data` | `model-data-pass` | Pinned model structure, tokenizer, target modules, every canonical row, trainable scope |
| 3 | `aptus run BUNDLE --action preflight` | `measured-preflight-pass` | Selected synthetic path, CUDA peak, census, placement bindings |
| 4 | `aptus run BUNDLE --action pilot` | `pilot-pass` | Both fresh phases, equal censuses, checkpoint continuation, measured peaks, pilot export |
| 5 | `aptus run BUNDLE --action train --confirm-full-train` | `measured-run-pass` after parent verification | Current admission, full metrics, split evidence, ranks, final export manifest |

Replace `BUNDLE` with the resolved bundle directory. Wait for each command to
reach a terminal state. Between actions, inspect:

```bash
aptus jobs
aptus jobs --id JOB_ID
```

The log path in the job record is authoritative. Preserve the complete log,
not only the returned tail.

## Dependency action

- [ ] Exact direct package versions are installed.
- [ ] The environment binding contains Python, platform, direct constraints,
      and runtime distribution closure.
- [ ] No unreviewed package substitution occurred.
- [ ] CUDA imports and driver/runtime compatibility pass where checked.

## Model-data action

- [ ] The exact model and tokenizer resolve at the pinned revision.
- [ ] Loaded parameter count and structural facts match within the documented
      contract.
- [ ] Every required adapter target module exists.
- [ ] The prepared method has a positive, finite, correctly scoped trainable
      set.
- [ ] Every canonical row transforms with non-empty supervision.
- [ ] Credentials remained in the underlying model stack, not the bundle.

Model-data validation does not enter training mode and does not prove CUDA fit.

## Measured preflight action

- [ ] The method, precision, quantization, distribution, world size, and device
      indices match the selected candidate.
- [ ] Measured peak CUDA bytes are positive.
- [ ] The trainable census is valid and bound to the selected method.
- [ ] Current unrelated GPU pressure is understood.

Preflight uses a synthetic method path. It does not load the real model and
data combination.

## Pilot action

- [ ] Both phases ran in fresh processes.
- [ ] Both phases bind the same plan, candidate, environment, hardware, and
      trainable census.
- [ ] Losses are finite and steps are positive.
- [ ] Phase one checkpoint evidence passes.
- [ ] Phase two proves the expected step continuation.
- [ ] Pilot artifact and export manifests pass.
- [ ] Measured CUDA peaks and checkpoint/export sizes are recorded.

Do not authorize training after any pilot failure. Preserve the failed metrics,
logs, and artifacts for diagnosis.

## Train admission

Immediately before submission:

- [ ] Confirm the expensive full-run action with the requestor.
- [ ] Confirm the current bundle manifest and plan identity.
- [ ] Confirm current CUDA identities and free VRAM.
- [ ] Confirm current free host RAM and disk.
- [ ] Confirm the installed environment still matches the pilot.
- [ ] Confirm pilot metrics, checkpoint trees, and pilot exports remain intact.
- [ ] Confirm the unique run output path does not exist.

The submission transaction performs these checks while holding the job locks
and host-global lease. Cached UI authorization is informational.

## Monitor training and verification

- [ ] Watch state, phase, log growth, device memory, host memory, disk, and
      unrelated workload changes.
- [ ] Treat `queued`, `running`, and `cancelling` as active.
- [ ] Treat phase `verifying` as parent-owned work within `running`.
- [ ] Do not infer success from a zero child exit.
- [ ] Do not edit or move the active run directory.
- [ ] Do not start another Aptus accelerator action under a different state
      root.

The parent must verify metrics, ranks, trainable census, dataset-split evidence,
and the structural export file tree before completion.

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
      ranks, and measured peaks.
- [ ] `final-export.json` covers the expected structural file tree.
- [ ] The unique run path, reports, logs, environment record, and manifest are
      retained together.
- [ ] Any quality statement is withheld until a separate evaluation passes.

## After failure or interruption

1. Preserve the job JSON, full log, report, plan, manifest, and run directory.
2. Classify the cause as dependency, model/data, method scope, CUDA capacity,
   distribution, checkpoint, split, export, cancellation, or parent
   verification.
3. Correct the environment, facts, data, or source implementation.
4. Recompile to a new bundle when compiler-managed content changed.
5. Repeat every invalidated action.
6. Submit a new full run with a new run ID.

Do not resume an arbitrary full checkpoint. Aptus 0.2 does not bind the complete
state needed for general full-run resume.

## Retention and cleanup

Aptus 0.2 has no automatic retention or cleanup policy. Establish one before
large runs. Never remove evidence needed to interpret an active job or an
artifact still in use. Record any manual deletion by exact path and retention
rule. Protect all deleted material according to its sensitivity and recovery
requirements.

## Related documentation

- [Compile, validate, and run](../guides/compile-validate-run.md)
- [Inspect results](../guides/inspect-results.md)
- [Recovery and the resume boundary](../guides/resume-recover.md)
- [Security policy](../../SECURITY.md)
- [Release gates](release-gates.md)
