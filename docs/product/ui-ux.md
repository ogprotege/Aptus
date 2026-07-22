# UI and UX Contract

The workbench is a local operator interface for the same API contracts exposed by
the CLI. It must make evidence state and blocked actions visible.

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
3. Run measured synthetic preflight.
4. Run the two-phase real-model and data pilot.
5. Confirm and start full training.

The UI must not present these as one unreviewed automatic action.

## Facts stage

Every editable fact has a clear unit and source. Optional model inspection shows
the raw provider-declared model type and architectures, fills supported
provider-declared facts, and can infer a normalized Aptus family through an
explicit alias rule. The UI distinguishes that inferred family from the raw
provider evidence and shows warnings. Inspection cannot check the
training-permission box or decide a license. Hardware scanning clearly names the
server host. Apple Silicon scans label the shared unified-memory pool and state
that the current compiler remains fail-closed. CUDA previews explain that
single-device rows bind the strongest method-compatible visible GPU, while
distributed rows use limiting VRAM and capabilities shared by every
participating GPU.

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
