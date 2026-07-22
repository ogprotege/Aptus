# Choose Your Aptus Path

> **Status:** Active | **Audience:** First-time users | **Authority:** Explanatory | **Applies to:** Aptus 0.2 | **Owner:** Product | **Last reviewed:** 2026-07-22 | **Review by:** 2026-10-22

Aptus can profile data, compare plans, compile bundles, and run static checks on
an ordinary development computer. Its current training compiler executes only
on CUDA. Choose a path based on the result you need and the host you actually
have.

## Choose by outcome

| I want to | Start here | Host requirement | Stop when |
|---|---|---|---|
| Understand Aptus without training | Open Aptus for Mac or start the browser workbench | macOS app or Python 3.11 or newer | You can explain the five stages and the evidence boundary |
| Profile a dataset | Run `aptus profile` | Any supported Python host | You have reviewed counts, schemas, duplicates, truncation warnings, and the source digest |
| Compare plans for a future CUDA host | Enter explicit, user-attested CUDA facts | Any supported Python host | You have reviewed every candidate status, assumption, and memory envelope |
| Produce a reviewable bundle | Compile a persisted plan and run static validation | Any supported Python host | The no-clobber bundle and archive pass `static-pass` |
| Validate and train | Run the five ordered actions on the target CUDA host | Compatible CUDA devices, dependencies, model access, and sufficient storage | Parent verification promotes the exact run to `measured-run-pass` |
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
5. Enter measured or user-attested facts for the intended CUDA host.
6. Compare the 12 method-placement candidates.
7. Compile the selected plan to a new path.
8. Run static validation.

Planning against manual CUDA facts is a forecast for another host. It is not a
measurement of the local computer and does not authorize training. Preserve the
`user-attested` provenance label.

On Darwin arm64, hardware inspection records one `mps` shared unified-memory
device when CUDA is absent. That inventory does not create an executable MPS or
MLX candidate. Aptus 0.2 does not convert unified memory into dedicated VRAM or
invent a current-free-memory value.

## Path B: Validate and train on CUDA

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

No real CUDA pilot has been completed on the current development Mac. The
repository remains an engineering preview until the applicable
[release gates](../operations/release-gates.md) pass on target hardware.

## Path C: Use the Mac app, browser workbench, or CLI

All three interfaces use the same Python contracts.

Choose the Mac app for a native launch surface, file and folder pickers,
authenticated private backend lifecycle, Finder actions, and explicit CUDA-host
handoff. It never submits CUDA work on macOS, even when the plan describes a
different CUDA machine.

Choose the workbench when you want guided fact entry, candidate cards, evidence
disclosures, and job monitoring. Keep it on loopback because the service can
read files and launch processes and has no authentication boundary.

Choose the CLI when you need repeatable local commands, persisted JSON, explicit
paths, or automation around a single-user host. The CLI does not weaken any
planner, compiler, validation, lease, or confirmation rule.

The portable bundle is a third interface. It carries its own validators and
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
- a measured preflight or pilot fails;
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
