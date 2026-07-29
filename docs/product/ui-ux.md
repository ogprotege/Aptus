# UI and UX Contract

> **Status:** Active | **Authority:** Normative interface contract | **Applies to:** Aptus 0.2 | **Audience:** Workbench contributors and reviewers | **Last reviewed:** 2026-07-27 | **Review by:** 2026-10-27 or when the workbench changes

The Mac product and contained workbench are local operator interfaces for the
same Python contracts exposed by the CLI. They must make runtime identity,
evidence state, and blocked actions visible.

## Native Mac shell

AppKit owns application lifecycle and the main window. SwiftUI owns the Home,
Workbench, Machine, and Models destinations. The shell uses system
typography and adaptive colors, with macOS 26 as the primary visual design and
macOS 15 as the deployment fallback.

The Machine destination reports measured Apple facts without promising model
fit. The Models destination keeps training runtimes separate from inference
services. Its `Choose MLX Python` action validates one exact executable through
the authenticated local API and displays the persisted absolute command path.

The complete React workbench appears inline inside the Workbench destination.
It owns one Facts, Compare, Compile, Validate, and Run workflow. Its complete
bridge provides dataset selection, output selection, Finder reveal, and
readiness reporting. Partial bridges are ignored.

Project history is part of this workflow. It lists immutable revisions, loads
revision detail on demand, and recovers an older state only by creating a new
revision. The interface must state that recovery does not restore training
authorization and must require fresh validation and confirmation.

Saved v2 plans and plans with no schema identifier remain historical records,
not executable workspaces. Bootstrap exposes `replan_required` and the source
identity. The workbench shows that message, restores no old plan or bundle, and
does not offer compile or revision recovery for it. The operator creates a new
deterministic v3 plan from the preserved facts. The UI must never imply that
changing the old schema label is a migration.

The Models destination includes a read-only MLX environment doctor. Each likely
interpreter shows path, discovery source, Python version, import-probe status,
and exact-pin compatibility. Only a compatible row can invoke **Use this
Python**, and the backend rechecks the contract before persistence. When
none pass, the interface shows the exact external virtual-environment recipe.
No doctor action installs or changes packages.

## Five stages

1. **Facts:** enter model, dataset, hardware, and target facts.
2. **Compare:** inspect feasible, conditional, infeasible, and unsupported
   candidates.
3. **Compile:** choose a new bundle path and create portable artifacts.
4. **Validate:** inspect the evidence ladder and run static validation.
5. **Run:** perform the five ordered runtime actions and monitor the current job.

The run stage contains five distinct actions:

1. Install and verify dependencies.
2. Resolve model and validate every canonical data row.
3. Run the runtime-specific measured preflight.
4. Run the runtime-specific real-model and data pilot.
5. Confirm and start full training.

The UI must not present these as one unreviewed automatic action.

## Facts stage

Every editable fact has a clear unit and source. Optional model inspection shows
the raw provider-declared model type and architectures, fills supported
provider-declared facts, and can infer a normalized Aptus family through an
explicit alias rule. The UI distinguishes that inferred family from the raw
provider evidence and shows warnings. Inspection cannot check the
training-permission box or decide a license. Hardware scanning clearly names the
server host. Apple Silicon scans label the shared unified-memory pool, select
the MLX-LM training runtime, keep device free VRAM unknown, use measured free
host RAM as live unified-memory headroom when available, and apply an 8 GiB
minimum reserve for local planning. CUDA previews explain that
single-device rows bind the strongest method-compatible visible GPU, while
distributed rows use limiting VRAM and capabilities shared by every
participating GPU.

When inspection returns MoE topology, the Facts stage shows a static expert
routing rail from token to router to the selected expert bank. It displays
experts selected per token, total experts, optional shared-expert presence,
sparse-layer count, expert width, checkpoint precision, total resident
parameters, and backend-derived active parameters. It also displays the exact
runtime, method, placement, and pilot boundary from the compatibility result.
The rail presents support only for a structurally complete conditional result.
Malformed or contradictory evidence renders a fail-closed unsupported state.
Changing any inspection-derived model identity or shape fact clears the topology
until the operator inspects again. Changing total parameters or training
permission does not erase inspection because those facts remain user-attested.
Changing total parameters clears any previously derived active-parameter value
until the backend creates a new plan.

The method preference is populated from the API's selectable registry entries.
The readiness board uses a separate status lane for gated executable,
experimental, and research-only identities. Every unavailable method states the
missing proof. Presence in that board never makes a method selectable.

Profiling statistics can be sampled, but the UI explains that compilation writes
every canonical row.

## Compare stage

Candidate cards show method, distribution, precision, quantization, exact batch
arithmetic, status, reasons, point estimate, upper envelope, available memory,
host RAM, disk, assumptions, and evidence. Unsupported rows remain visible.
Selecting a row changes only the inspected evidence. Compilation always uses the
plan's clearly labeled recommended candidate.

The recommendation label means highest-ranked within the enumerated viable
catalog. Viable includes `feasible` and `conditional`, with feasible ranked
first. A conditional label keeps its unresolved warning. Recommendation does
not mean “best model” or “guaranteed fit.”

## Compile and validate stages

Compilation requires a new path and explains no-clobber behavior. The artifact
view lists the bundle and archive. Validation presents each evidence level, its
bindings, findings, and the difference between analytic and measured evidence.

The dependency file is labeled as direct exact pins, not a transitive lock.

## Run stage

The next forward action is selected and emphasized automatically. Earlier
passed actions remain available for an explicit recheck, while forward skips
stay disabled. The UI shows job ID, action, state, phase, log location,
timestamps, errors, and cancellation state. During parent verification it
displays `verifying`, not a premature success.

Train confirmation states that the action can consume substantial compute and
write checkpoints and model artifacts. Full resume is not offered.

Pilot success can be shown before train admission, but the UI must explain that
deep authorization occurs when train is submitted. Cached authorization text is
not treated as a durable entitlement.

For an MLX-LM bundle on the Mac, the run stage exposes local dependency,
model-data, measured-preflight, pilot, and confirmed full-training jobs in their
required order. Pilot status must explain its uninterrupted two-update run and
fresh-process adapter generation. Full training becomes available only after
current `pilot-pass` evidence. The UI must never label MLX weight snapshots as
resumable checkpoints.

For a CUDA bundle, the Mac app presents target-host handoff instead of local
run controls. This remains true when the operator enters manual CUDA facts for
another host. PyTorch MPS must not appear executable until it has a compiler.

LM Studio and oMLX controls, when present, must say inference or evaluation.
They must never satisfy a training-runtime field or evidence gate.

On completion, display the unique run output directory, report state, completion
attestation, final export manifest location, and artifact-integrity status. Label
the export check as structural. Do not imply benchmark quality.

## Active-job behavior

Facts, hardware scan, local-scan planning, compilation selection, and competing
runtime actions are blocked as appropriate while a managed job is active.
Polling must remain cheap. Cancellation remains available until the parent enters
its non-cancellable completion commit.

## Accessibility and responsive behavior

- Every field has a programmatic label and useful error text.
- Focus moves to the selected stage heading.
- Status changes use a polite live region; blocking errors use an alert.
- Color is not the only status signal.
- Keyboard users can reach every action and disclosure.
- Narrow layouts preserve stage order, evidence labels, and action controls.
- Motion respects reduced-motion preferences.

## Example mode

Example data is labeled on every relevant screen. It must never look like a real
inspection, plan, compile, validation, pilot, or run result.

The canonical visual tokens and components live in `web/src/styles.css` and
`web/src/components/`.

## Related documentation

- [Workbench development](../contributing/workbench.md)
- [Current capabilities](current-capabilities.md)
- [Validation states](../reference/validation-states.md)
- [Claim language](claim-language.md)
