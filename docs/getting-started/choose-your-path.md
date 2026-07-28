# Choose Your Aptus Path

> **Status:** Active | **Audience:** First-time users | **Authority:** Explanatory | **Applies to:** Aptus 0.2 | **Owner:** Product | **Last reviewed:** 2026-07-27 | **Review by:** 2026-10-27

Aptus can profile data, compare plans, compile bundles, and run static checks on
an ordinary development computer. Its CUDA compiler covers the complete
evidence ladder. Its separate MLX-LM compiler on Apple Silicon covers
uninterrupted LoRA and QLoRA pilot and full-duration adapter training. Choose a
path based on the result you need and the host you actually have.

## Choose by outcome

| I want to | Start here | Host requirement | Stop when |
|---|---|---|---|
| Understand Aptus without training | Open Aptus for Mac or start the browser workbench | macOS app or Python 3.11 or newer | You can explain the five stages and the evidence boundary |
| Profile a dataset | Run `aptus profile` | Any supported Python host | You have reviewed counts, schemas, duplicates, truncation warnings, and the source digest |
| Compare plans for an Apple or CUDA host | Select the training runtime and enter measured or user-attested facts | Any supported Python host | You have reviewed every candidate status, runtime contract, assumption, and memory envelope |
| Produce a reviewable bundle | Compile a persisted plan and run static validation | Any supported Python host | The no-clobber bundle and archive pass `static-pass` |
| Validate and train an MLX-LM bundle | Run the five ordered actions | Apple Silicon, the exact external MLX-LM Python, model access, and sufficient storage | Parent verification promotes the exact uninterrupted adapter run to `measured-run-pass` |
| Validate and train on CUDA | Run the five ordered actions on the target CUDA host | Compatible CUDA devices, dependencies, model access, and sufficient storage | Parent verification promotes the exact run to `measured-run-pass` |
| Change Aptus | Use the contributor path | Python, Node.js, and the repository checkout | The relevant Python, web, packaging, and documentation checks pass |

## Path A: Learn and plan locally

Use this path on macOS, a CPU-only machine, or any host that is not the final
CUDA training host.

1. [Build Aptus for Mac](install.md#build-aptus-for-mac), install Aptus for
   Python, or both.
2. Open the Mac app. On another platform, start the browser workbench with
   `aptus serve --host 127.0.0.1 --port 8787`, or use the CLI.
3. Profile a local dataset.
4. Inspect the local hardware inventory if useful.
5. Select MLX-LM for Apple Silicon or Transformers and PEFT for CUDA.
6. Enter measured or user-attested facts for the intended host.
7. Compare the 12 method-placement candidates and their runtime contracts.
8. Compile the selected plan to a new path.
9. Run static validation.

Planning against manual hardware facts is a forecast for another host. It is
not a measurement of the local computer and does not authorize training.
Preserve the `user-attested` provenance label.

On Apple Silicon, hardware inspection records one `mps` shared unified-memory
device. Aptus does not convert unified memory into dedicated VRAM or copy host
free RAM into free VRAM. MLX-LM uses a separate unified-memory estimator.
PyTorch MPS remains a known runtime without a compiler.

## Path B: Run an MLX-LM bundle on Apple Silicon

Use this path on the Mac that will run the measured checks.

1. Build and open Aptus for Mac.
2. Create an external Python environment with the pinned MLX and MLX-LM
   versions.
3. Open **Models**, choose **Choose MLX Python**, and select that environment's
   exact Python executable.
4. Open the contained workbench and scan this Mac.
5. Select MLX-LM and a single-device LoRA or QLoRA candidate.
6. Compile to a new bundle path and pass static validation.
7. Run dependency validation.
8. Run model-data validation. Aptus loads the pinned revision and tokenizes all
   bound train and validation rows.
9. Run measured preflight. Aptus runs a bounded adapter smoke and records
   runtime-neutral memory metrics.
10. Run pilot. Aptus trains from the pinned base without interruption for at
    least two optimizer updates, verifies finite losses and exact target
    coverage, then reloads the adapter in a fresh process for one to four tokens.
11. Inspect `pilot-pass` and current headroom admission.
12. Confirm full training only when the requestor accepts its cost. The full run
    starts again from the pinned base and runs for the plan-derived duration.
13. Wait for parent verification and `measured-run-pass`.

MLX-LM crash resume is unsupported. Its periodic files are weight snapshots,
not resumable checkpoints, and every resume argument fails. QLoRA also requires
a pinned MLX model with explicit four-bit quantization metadata. It never uses
bitsandbytes.

## Path C: Validate and train on CUDA

Use this path only on the target host named by the plan.

1. Confirm the immutable model revision, license label, and permission to
   train.
2. Protect the dataset, bundle, archive, model cache, logs, checkpoints, and
   exports as sensitive files.
3. Install the bundle's exact direct pins in an isolated environment outside
   the bundle directory.
4. Run dependency validation.
5. Run model-data validation.
6. Run measured preflight.
7. Run the two-phase real-model pilot.
8. Review the current admission evidence.
9. Start full training with explicit confirmation.
10. Inspect the parent-verified result.

Do not skip an action. A higher validation action repeats its lower checks, but
the managed workflow still requires the preceding recorded state before it
admits a forward action.

No real CUDA pilot has been recorded for this release. The repository remains
an engineering preview until the applicable
[release gates](../operations/release-gates.md) pass on target hardware.

## Path D: Use the Mac app, browser workbench, or CLI

All three interfaces use the same Python contracts.

Choose the Mac app for the AppKit lifecycle, SwiftUI Home, Workbench, Machine,
and Models shell, exact MLX Python selection, authenticated private backend,
Finder actions, local MLX-LM gates, and explicit CUDA-host handoff. The full
React workbench is inline and owns one Facts, Compare, Compile, Validate, and Run
workflow with immutable project history.

Choose the workbench when you want guided fact entry, candidate cards, evidence
disclosures, and job monitoring. `aptus serve` prints a new authenticated
workbench URL and bearer token on every launch. Open the printed URL so its
one-time query handoff can set the HttpOnly, SameSite Strict cookie and redirect
to a clean URL. Keep the service on loopback because its protected API can read
files and launch processes. The token is single-user access control, not tenant
isolation.

Choose the CLI when you need repeatable local commands, persisted JSON, explicit
paths, or automation around a single-user host. The CLI does not weaken any
planner, compiler, validation, lease, or confirmation rule.

The portable bundle is an additional interface. It carries its own validators and
parent runner. Direct portable full-run execution is supported on POSIX. On
Windows, use the managed `aptus run` path because direct portable child process
control is fail-closed in Aptus 0.2.

## Know when to stop

Stop and correct the facts or implementation when any of these occurs:

- no candidate is feasible or conditional;
- a source digest changes after profiling;
- the loaded model does not match the pinned structural facts;
- target modules or the trainable-parameter census do not match the selected
  method;
- a measured preflight or runtime-specific pilot fails;
- current VRAM, host RAM, or disk no longer passes admission;
- the final metrics or export tree fails parent verification.

Do not edit a compiled bundle or validation report to bypass a finding. Replan
and compile to a new path when compiler-managed content changes.

## Related documentation

- [Quickstart](quickstart.md)
- [Current capabilities](../product/current-capabilities.md)
- [Model, dataset, and hardware facts](../guides/model-dataset-hardware.md)
- [Compile, validate, and run](../guides/compile-validate-run.md)
- [Security boundaries](../architecture/security-boundaries.md)
