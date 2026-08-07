# Aptus Product, Architecture, Refactor, and Interface Review

> **Documentation status:** Archived and superseded review evidence
>
> **Applies to:** Point-in-time product and architecture review recorded below
>
> **Last reviewed:** 2026-08-06
>
> **Next scheduled review:** 2027-08-06, or when provenance or a named successor changes
>
> **Historical warning:** This review is preserved without rewriting its body.
> Statements below that say a condition is current, open, or complete describe
> the reviewed snapshot, not the present repository. Use the
> [historical-review index](../README.md) to find current successors.

> **Status:** Superseded implementation baseline, retained as decision evidence
> **Superseded on:** 2026-07-27
> **Current sources:** `docs/product/current-capabilities.md`,
> `docs/operations/release-gates.md`, and the dated MLX-LM acceptance record

The critical findings below describe the pre-implementation snapshot. The
approved implementation has since corrected shutdown containment, unified the
native and web product surface, added immutable project history, completed two
clean MLX-LM workflows through `measured-run-pass`, added typed API and OpenAPI
contracts, and strengthened packaging. The unchecked boxes remain historical
review text. They do not state current product status. CUDA target-host evidence
and public notarization remain open.

**Last updated:** 2026-07-27
**Review type:** Read-only product and code review
**Decision at review time:** Prove one real run, correct desktop shutdown
semantics, and unify the product surface before a broad refactor. Those three
conditions are now implemented and independently documented.

## Executive assessment

Aptus has the bones of an exceptional product. Its evidence model is unusually
disciplined. The planner, compiler, validator, authenticated desktop sidecar,
and fail-closed runtime gates form a serious technical core. The repository has
extensive tests. The React workbench already carries a distinct visual identity.

Aptus is not yet a complete fine-tuning product. It is a strong planner and
bundle compiler with an unfinished operator experience. Four facts control the
next release:

1. No real Apple Silicon or CUDA pilot has passed all release gates.
2. Desktop shutdown can report completion with survivors, and its release test failed intermittently.
3. The complete workflow sits inside a modal sheet behind a second navigation shell.
4. The first stage asks users for low-level facts before it establishes their intent.

Adding more methods would not solve these problems. A magnificent Aptus should
make one difficult promise and prove it: a user can bring a real model and data
to a measured, reviewable, reproducible training result without hidden guesses.

## What should remain

- Keep Python as the sole authority for plans, candidates, validation, and jobs.
- Keep estimates, measurements, and quality evidence as distinct categories.
- Keep unsupported candidates visible and explained.
- Keep explicit training permission and license attestations.
- Keep no-clobber output, exact identity binding, and parent-side verification.
- Keep MLX-LM separate from MPS, CUDA, LM Studio, and oMLX.
- Keep the Fit Ledger as the signature product element.
- Keep the current adaptive teal, amber, and evidence-instrument visual language.
- Keep native code focused on lifecycle, security, paths, system facts, and Mac integration.

## Critical findings

### [ ] C1. The central execution claim lacks target-host proof

**Evidence:** `README.md:28-34` states that no real CUDA or Apple Silicon pilot
has completed the release gates. `docs/operations/release-gates.md` treats a
real target-host pilot as release evidence, not an optional demonstration.

**Why it matters:** Aptus can prove that it planned, compiled, and statically
validated a bundle. It cannot yet prove that its primary local training path
works through the complete product on real hardware. This limits honest product
language to an engineering preview.

**Required correction:** Establish one canonical Apple Silicon acceptance run.
Use a small public model pinned to an immutable revision and a repository-owned
synthetic dataset. Capture environment identity, model resolution, canonical
data validation, measured preflight, the required uninterrupted pilot, adapter
reload, and a bounded confirmed training run. Preserve logs, hashes, output
identity, memory measurements, timing, and the exact Aptus commit.

**Acceptance gate:** A clean checkout on the supported Mac completes the same
recorded workflow twice without manual file repair or undocumented shell steps.

### [ ] C2. Desktop shutdown can complete while a descendant remains

**Evidence:** The first `desktop/macos/build.sh` run failed
`BackendControllerIntegrationTests.testStopForceTerminatesATermResistantProcessTreeBeforeCompleting`
at `desktop/macos/Tests/BackendControllerIntegrationTests.swift:180-207`. The
child process still appeared alive when the stop completion ran. A targeted
rerun passed. A second complete build also passed.

`BackendController.swift:253-348` repeatedly snapshots, suspends, and terminates
the process tree. The observed failure does not prove whether PID identity,
zombie reaping, fixture timing, or shutdown ordering caused that test result.

The timeout path is independently unsafe. When tracked processes remain after
the forced deadline, `evaluateShutdown()` calls `finishShutdown(failure:)`
(`BackendController.swift:314-323`). `finishShutdown()` drops process and path
ownership and runs every void stop callback (`BackendController.swift:334-370`).
`restart()` starts a replacement from that callback
(`BackendController.swift:111-118`). Application termination also treats the
same callback as permission to quit (`AptusApplication.swift:3-17`, `45-52`).

**Why it matters:** Aptus makes strict claims about child-process containment.
The timeout path can abandon a surviving sidecar or descendant, remove the
session record, and then permit a replacement start or application exit. This
violates the ownership contract even if the intermittent test has a separate
fixture or process-state cause.

**Required correction:** Give shutdown a typed success or failure result. Never
run restart or successful-termination callbacks after a timeout with living
targets. Retain enough ownership to report and retry cleanup. Add diagnostic
state before changing timing constants. Record PID identity, process state,
parent PID, signal results, and termination-handler order. Replace wall-clock
assumptions with an observable completion condition.

The current test helper uses `kill(pid, 0)`, which cannot distinguish a zombie
(`BackendControllerIntegrationTests.swift:413`). `BackendProcessIdentity` does
not retain process state (`BackendProcessTree.swift:4-10`). The corrected path
must make its successful-stop condition observable before relying on it.

**Acceptance gate:** Injected timeout tests prove that restart does not launch a
replacement and termination does not report success while targets remain. Then
1,000 isolated shutdown iterations and 10 consecutive full desktop builds pass
on the release host. Successful stop must imply that every captured non-zombie
descendant has disappeared and the session directory is gone.

## Important findings

### [ ] I1. Aptus presents two shells for one workflow

**Evidence:** `DesktopShell.swift:192-300` defines Home, Machine, Models, Data,
Plans, and Runs navigation. Data, Plans, and Runs mostly direct users to the
workbench (`DesktopShell.swift:780-844`). The full React workflow opens in a
modal sheet with a fixed minimum size (`DesktopShell.swift:1053-1067`). React
then presents its own five-stage rail (`web/src/App.tsx:745-886`). The current
architecture calls this workbench transitional
(`docs/product/ui-ux.md:21-24`).

The placeholder shell has already drifted from the real contract. Its Plans
sequence displays Validate before Compile (`DesktopShell.swift:805-813`), while
the React rail and product contract correctly place Compile before Validate
(`WorkflowRail.tsx:8-14`).

**Why it matters:** The shell looks finished while the useful product is hidden
one level deeper. Users must learn two navigation systems. Native Plans and Runs
suggest persistent product areas but act as signposts to the modal workflow.

**Correction:** Make the workbench the primary detail surface in the native
window. Retain one global sidebar for projects, runtimes, and history. Place the
five stages inside the selected project. Remove the modal as the normal entry
point. Keep the bridge narrow and keep domain logic in Python.

**Why this route:** A full Swift rewrite would consume time without proving more
training behavior. The existing React flow is complete, tested, and visually
distinct. Embed it coherently, then replace isolated surfaces only when native
code gives a clear Mac-specific benefit.

### [ ] I2. Facts intake starts at the expert schema, not the user's goal

**Evidence:** `FactsStage.tsx:208-570` presents four large panels. The model
panel asks for family, parameter count, hidden size, layers, context length,
intermediate size, license, and permission. Hardware and target panels expose
more implementation values before a plan exists. Inspection fills some fields,
but parameter count, license judgment, and permission correctly remain explicit
(`FactsStage.tsx:215-308`).

**Why it matters:** The form reflects the domain model accurately but makes the
user act as Aptus's integration layer. A new user can possess a valid model and
dataset yet still fail before seeing the product's value.

**Correction:** Use a guided intake with three initial choices:

1. Select or paste a model reference.
2. Select a dataset.
3. State the intended task and constraint.

Inspect the model, scan the local Mac, and profile the data automatically after
explicit selection. Show unresolved facts as a review queue. Put architecture,
memory reserve, batch arithmetic, and provider evidence in an Advanced Facts
drawer. Keep license and training permission as required human attestations.

### [ ] I3. Generated runtimes are embedded inside a 7,545-line compiler file

**Evidence:** `src/aptus/generation.py` contains raw executable constants at
lines 27, 3169, 3847, 4384, 4533, 4559, 5417, 5602, and 6047. Compiler functions
follow near line 6707.

**Why it matters:** The embedded programs contain normal imports and already
receive rendered AST and focused execution tests. As giant strings, however,
they cannot receive ordinary pre-render formatting, linting, type analysis, or
safe symbol refactoring. The compiler also becomes the review bottleneck for
both CUDA and MLX-LM changes.

**Correction:** Prefer valid `.py` package resources copied byte for byte, since
most generated programs read user facts from `plan.json`. Use a template only
where source interpolation is truly required. Give CUDA and MLX-LM separate
directories. Preserve generated bytes and manifest hashes during the first
move. Add pre-render linting plus current snapshot and execution-contract tests.
Declare the resources in setuptools and PyInstaller, both of which currently
collect only existing web assets (`pyproject.toml:42`, `AptusBackend.spec:27`).

### [ ] I4. Core orchestration files carry too many responsibilities

**Evidence:** `execution.py` is 3,170 lines, `validation.py` is 1,748,
`api.py` is 1,046, `App.tsx` is 888, and `DesktopShell.swift` is 1,131.
`App.tsx:233-255` owns twenty state values and three refs. `api.py:309-1027`
composes middleware, errors, and every endpoint. `execution.py:1773-3170` puts
the job service after a large block of verification functions.

**Why it matters:** These files make behavior easy to couple and hard to review.
They also encourage private cross-boundary imports. For example,
`validation.py:38` imports `_actual_hardware_binding` from `execution.py`, and
`ApiContext` calls execution's private `_atomic_write_json`
(`api.py:212-230`). Environment-binding algorithms also appear separately in
validation, execution, and emitted runtime source (`validation.py:112`,
`execution.py:1652`, `generation.py:6380`).

**Correction:** Split by stable responsibility and keep compatibility facades.
Extract public host-side storage and binding primitives, then add parity tests
between host and portable binding behavior. Do not move every file in one
change. The migration plan below defines the order.

### [ ] I5. Users cannot manage durable named projects or revisions

**Evidence:** The React app owns the active workflow in component state
(`App.tsx:233-252`), but bootstrap restores the latest validated plan, bundle,
facts, report, and job (`App.tsx:284-340`). The API validates the saved bundle
reference before restoring it (`api.py:575-718`). Native Plans and Runs do not
expose a project library or revision history (`DesktopShell.swift:794-844`). No
named project aggregate owns facts, decisions, artifacts, and runs together.
The plan name only suggests an output directory and is absent from the plan
request (`FactsStage.tsx:194-204`, `api.ts:97-159`). A restored workspace
therefore falls back to its plan ID (`App.tsx:109-110`).

**Why it matters:** Fine-tuning is not a single-session form submission. Users
must compare attempts, resume review, recover artifacts, and understand which
facts produced a run.

**Correction:** Build on the existing latest-session restore. Add a local named
project manifest with immutable revisions. A project
should own fact snapshots, plan IDs, selected decisions, bundles, validation
reports, and jobs. Load the most recent safe revision on relaunch. Never infer
training authorization from an old project revision. Use atomic writes, private
permissions, symlink rejection, referential-integrity checks, corrupt-record
quarantine, and explicit import from current `plans/`, `current-bundle.json`,
and job records.

### [ ] I6. The embedded comparison layout responds to the wrong width

**Evidence:** The candidate table has a `1080px` minimum width
(`styles.css:1762-1771`). The card layout activates at a viewport media query of
`920px` (`styles.css:2615`). At the 1,040-point workbench sheet width, the
1,160-point breakpoint removes the Fit Ledger inspector, but the 920-point card
breakpoint remains inactive. React's 224-point rail and padding then leave the
inner work area far below the table's minimum width.

**Observed result:** At the sheet's 1,040 by 700 minimum, the comparison requires
horizontal scrolling. The sticky action area transiently occludes lower content
during scrolling at that size, even though it remains in document flow and the
Facts form reserves bottom space (`styles.css:676-678`, `1303-1318`).

**Correction:** Base comparison mode on the main content container, not the
WebView viewport. Use a container query or a layout measurement. Keep action
controls in document flow or reserve an explicit safe area. Test widths after
subtracting the native sidebar, React rail, inspector, and padding.

### [ ] I7. Public Mac distribution remains incomplete

**Evidence:** `README.md:32-34` and `desktop/macos/README.md` state that the app is
locally signed and not notarized.

**Why it matters:** A source build and a CI artifact serve reviewers. They do
not provide a normal installation path for the intended Mac user.

**Correction:** Add Developer ID signing, notarization, stapling, quarantine-path
installation, first-launch, sidecar, and update-strategy gates. Keep review
binaries in CI artifacts rather than Git.

### [ ] I8. Current documentation contradicts current capability

**Evidence:** `docs/product/vision.md:41-42` calls Aptus 0.2 the local CUDA core
and says later integrations are absent. Current code and other active documents
include MLX-LM, local inference adapters, and a native Mac product. The older
review under `dev/active/aptus-macos/` also records stale macOS and test facts.

The contradiction also changes the documented security boundary.
`SECURITY.md:32-49` says ordinary `aptus serve` has no authentication and the
desktop sidecar rejects job submission. The CLI creates a random session token
(`cli.py:564-580`). The desktop starts `create_app(..., execution_enabled=True)`
(`desktop.py:94-99`).

**Correction:** Update the canonical vision and mark the older review as
superseded. Correct the security document before calling the current desktop
boundary reviewed. Keep historical evidence but do not let it appear current.

### [ ] I9. The Fit Ledger uses CUDA language for Apple unified memory

**Evidence:** `FitLedger.tsx:36-45` labels the available amount as usable
per-device VRAM for every candidate. The Apple path correctly describes one
shared unified-memory pool in `FactsStage.tsx:362-365` and
`docs/product/ui-ux.md:53-56`. Candidate transport data already includes a
runtime contract (`types.ts:201-209`).

**Why it matters:** VRAM language suggests a separate GPU pool on Apple Silicon.
That conflicts with Aptus's strongest platform distinction and can mislead the
user about fit.

**Correction:** Choose labels and explanations from the candidate's runtime
contract. Use unified-memory headroom for MLX-LM. Use device VRAM for CUDA. Keep
host staging memory separate in both cases.

### [ ] I10. Candidate selection looks executable but only changes inspection

**Evidence:** The comparison invites users to select a strategy and applies an
`aria-pressed` state (`CandidateComparison.tsx:36-43`, `69-79`). Compilation
still uses the planner's recommended candidate (`CompareStage.tsx:199-207`,
`App.tsx:553-565`). The normative contract confirms that row selection changes
only inspected evidence (`docs/product/ui-ux.md:71-75`).

**Why it matters:** The visual selection implies that the chosen row will be
compiled. A user can reasonably believe they overrode the recommendation when
they did not.

**Correction:** Rename the action to Inspect and reduce selected-plan styling.
If users may influence the plan, provide a separate preference change that
replans and produces a new decision record.

### [ ] I11. First-launch MLX runtime setup depends on finding an executable

**Evidence:** The native setup asks the user to locate an exact Python
executable, including inside hidden environment folders
(`DesktopShell.swift:129-149`). The backend validates the chosen interpreter,
which is the correct trust boundary.

**Why it matters:** A valid Aptus install can still appear nonfunctional to a
user who does not know where an MLX-LM environment keeps its Python executable.

**Correction:** Add an explicit environment doctor. Detect likely interpreters,
show why each passes or fails, let the user confirm one, and provide a bounded
installation recipe when none qualifies. Do not silently install or switch a
runtime.

### [ ] I12. Accessibility foundations lack an acceptance gate

**Evidence:** The workbench has visible focus, text-backed status, live regions,
stage-heading focus, and reduced-motion rules (`styles.css:84-95`, `2960-2968`,
`App.tsx:360-362`, `788-800`). Native code also respects Reduce Transparency
(`DesktopShell.swift:1113-1128`). The web package has no accessibility test
dependency (`web/package.json:20-30`).

**Correction:** Require keyboard-only completion, VoiceOver reading order and
names, 200 percent text zoom, Increased Contrast, light and dark appearances,
and automated contrast and semantics checks. Automation supplements manual
VoiceOver testing. It does not replace it.

### [ ] I13. Persisted job records have no schema version

**Evidence:** `JobService._read()` accepts a dictionary when its embedded ID
matches the filename (`execution.py:1913-1926`). Newly submitted records do not
carry a persistence schema version (`execution.py:2603-2645`).

**Why it matters:** Named projects and revision history would make these records
long-lived product data. Unversioned records leave no safe migration or
corruption policy when fields and invariants change.

**Correction:** Define a versioned local persistence envelope before adding the
project library. Add backward-compatible readers, explicit migrations,
quarantine for corrupt records, and tests across every supported version.
Training authorization must remain a fresh action, not migrated durable state.

### [ ] I14. API response contracts drift across Python, TypeScript, and Swift

**Evidence:** FastAPI endpoints return many dictionary responses. TypeScript
casts and normalizes them manually (`api.ts:332-410`). Swift separately decodes
the same contracts (`DesktopBackendClient.swift:274-420`). The documentation
debt log records missing stable response models (`documentation-debt.md:67`).

**Why it matters:** One backend change can silently diverge in two clients. A
project and history feature would expand that risk.

**Correction:** Define explicit response models first. Check in and version the
OpenAPI artifact second. Generate TypeScript transport types, and either
generate Swift transport types or verify its deliberate Decodable models
against the same schema. Map transport types into client-specific view models.

## Minor findings

### [ ] M1. Product version is repeated across implementation surfaces

`0.2.0` appears in `pyproject.toml`, Python package code, API responses, request
headers, generated manifests, `web/package.json`, and the Mac `Info.plist`.
Create one release-time source and generate or verify every emitted copy.

### [ ] M2. TypeScript combines transport and UI-oriented types

`web/src/types.ts` has 617 lines. After I14 establishes stable transport types,
separate them from deliberate UI view models. Do not expose Python domain
internals directly to rendering components.

### [ ] M3. The build emits dependency-tooling warnings

The successful build emitted a Starlette deprecation warning for HTTPX client
compatibility and npm warnings about blocked install scripts for `esbuild` and
`fsevents`. Record and resolve them before they become release failures.

### [ ] M4. The old desktop review is no longer a safe status page

It reports older test totals and an obsolete macOS 13 floor. Add a supersession
banner rather than silently rewriting its dated conclusions.

## Security and safety posture

No new remote-code-execution finding emerged from this review. The desktop's
random session token, exact-origin WebKit policy, trusted-host check, private
paths, and explicit runtime admission form a good local boundary. The critical
shutdown finding still matters to security because an orphaned sidecar can
outlive the session ownership that should contain it.

The current desktop can submit execution jobs, so its authenticated API has
filesystem and subprocess authority. Correct `SECURITY.md` before expanding the
bridge or distribution. Keep dataset, bundle, log, cache, and model artifacts
classified as potentially sensitive cleartext. A public release also needs
Developer ID signing, notarization, and a tested quarantine path.

## Testing, performance, and maintainability

The test base is a strength. Characterization tests already cover generated
source, bundle identity, API behavior, job transitions, native process control,
and client flows. Refactoring should move behind these tests and add parity
checks before it changes behavior.

This review did not run a profiler or a real training workload. It therefore
makes no claim about UI latency, training throughput, memory fit, or energy use.
The oversized files are maintainability risks, not measured runtime bottlenecks.
Performance acceptance should use the canonical target-host run and record peak
unified memory, time per update, startup latency, artifact size, and UI polling
cost without turning those measurements into universal fit claims.

## Target architecture

### Dependency graph evidence

The generated repository graph contains 2,415 nodes, 5,970 undirected edges,
and 112 communities. Its most connected nodes include `JobService` with 86
edges, `BundleGenerationTests` with 52, `Backend` and `make_plan()` with 49
each, `BackendController` and `Method` with 40 each, and both
`validate_plan_payload()` and `validate_bundle()` with 36 each.

The graph therefore confirms two useful facts. Execution is the main
cross-community bridge, and tests already encode many of the contracts needed
for an incremental split. The graph health check also reported 373 dangling
endpoint edges and endpoint-pair collapse in the undirected view. Treat its
centrality as supporting evidence, not as proof of every dependency.

### Dependency rule

```text
domain + method contracts
          |
          v
profiling / inspection / planning
          |
          v
application services
          |
          +--> compiler writers ---> package programs --> bundle
          +--> validators ---------> evidence records
          +--> job service --------> runtime adapters + persistence
          |
          v
API routers / CLI commands
          |
          v
React client / native Mac client
```

Lower layers must never import HTTP, Swift, or React concepts. Validation must
not import a private helper from the job service. API and CLI surfaces should
call the same application services. Generated programs should not import from
the live Aptus checkout unless the bundle contract explicitly requires it.

### Proposed Python structure

```text
src/aptus/
  domain/
    facts.py
    plans.py
    evidence.py
    jobs.py
  application/
    profile_service.py
    plan_service.py
    compile_service.py
    validation_service.py
    job_service.py
  generation/
    writer.py
    manifest.py
    programs/
      cuda/{train,run,preflight,validate}.py
      mlx/{train,run,reload,preflight,validate}.py
  validation/
    artifact.py
    bindings.py
    static.py
    measured.py
    report.py
  execution/
    admission.py
    process.py
    verification.py
    persistence.py
    service.py
  api/
    app.py
    dependencies.py
    errors.py
    routers/{bootstrap,facts,plans,bundles,jobs,runtimes,inference}.py
```

This is a conceptual destination map, not an immediate filesystem layout. A
module such as `execution.py` cannot coexist with an importable `execution/`
package under the same name. Use temporary uniquely named internals such as
`aptus/_internal/execution/`, or perform one atomic namespace conversion with
re-exports from the new package. Broad imports across CLI and tests make a
pretend same-name facade unsafe. Do not split every domain dataclass or wrap
pure functions in services. Extract only cohesive seams with more than one
caller or independent change pressure.

`plan_contract.py` is intentionally self-contained because Aptus copies it into
portable bundles (`plan_contract.py:20`, `generation.py:7432`). Preserve that
boundary. Generated programs must not acquire an import dependency on the live
or installed Aptus package.

### Proposed React structure

```text
web/src/
  app/
    AptusApp.tsx
    workflowReducer.ts
    workflowEffects.ts
    projectSession.ts
  features/
    intake/
    comparison/
    compilation/
    validation/
    runs/
    runtimes/
  api/
    generated/
    client.ts
    mappers.ts
  design/
    tokens.css
    foundations.css
    components.css
```

Use a reducer or explicit state machine for legal workflow transitions. Keep
remote polling in focused hooks. Make busy state action-specific. Preserve the
existing complete-state behavior. No full-screen loader should replace usable
content when a local section can report progress.

### Proposed native structure

```text
desktop/macos/Sources/
  App/
  Backend/
  Platform/
  Workbench/
  Features/Home/
  Features/Runtimes/
  Features/Projects/
  Features/History/
  DesignSystem/
```

Split `DesktopShell.swift` by feature after the primary surface decision. Avoid
building native copies of planning and run state. Native feature models should
consume small API response types and system services.

## Safe migration order

### Phase A. Freeze behavior with characterization tests

1. Snapshot every generated file for representative CUDA and MLX-LM plans.
2. Record bundle tree, file bytes, modes, manifest fields, and deterministic ZIP hash.
3. Add contract tests for all current API routes and error envelopes.
4. Stress the desktop shutdown path and record diagnostics.
5. Add React tests for stale-response rejection, polling cancellation,
   bundle-report matching, and active-job mutation blocking.
6. Replace native source-path assertions with behavior tests before moving Swift files.

### Phase B. Extract generated programs without behavioral change

1. Move one raw runtime string into one valid Python package resource.
2. Copy it with the existing exact bytes.
3. Assert byte-for-byte parity with the old compiler output.
4. Repeat one action at a time.
5. Keep compiler entry points stable.
6. Run compilation from a source checkout, installed wheel, and frozen sidecar.

### Phase C. Separate execution and validation

1. Extract shared hardware identity into a public bindings module.
2. Move parent-side artifact verification into focused validators.
3. Move persistence, leases, subprocess control, and admission into separate modules.
4. Keep `JobService` as a compatibility facade until callers migrate.

### Phase D. Split transport composition

1. Extract FastAPI routers by resource.
2. Centralize dependencies, authentication, and error envelopes.
3. Define explicit response models and check in a versioned OpenAPI artifact.
4. Generate TypeScript transport types and verify Swift models against it.
5. Keep client view models independent from transport representations.

Router extraction must preserve one `ApiContext`, session middleware, static
catch-all ordering, and job-recovery side effects. These are application
invariants, not incidental placement.

### Phase E. Split the clients along product seams

1. Replace independent React state values with an explicit workflow reducer.
2. Extract effects and polling from rendering.
3. Split Facts into guided intake, evidence review, and advanced settings.
4. Split the native shell only after the unified window structure is accepted.

The reducer must preserve stale-response rejection, bundle-report identity,
active-job mutation blocking, and polling cancellation. Native splitting must
preserve the authenticated bridge and the two-part React readiness gate.

## Interface direction

### Subject and user

Aptus should feel like a calibrated scientific instrument for fine-tuning work,
not a generic analytics dashboard. Its primary user understands models but may
not know every runtime field. The interface should convert an intention into a
chain of inspectable evidence.

### Color system

Use six functional roles. The values below are dark-mode web mappings. Define
tested light counterparts. Native code should continue using semantic AppKit
colors so appearance, Increased Contrast, and accessibility settings remain
authoritative:

| Token | Suggested value | Role |
| --- | --- | --- |
| Obsidian | `#111719` | Window and deep background |
| Instrument | `#1B2528` | Main working surface |
| Evidence cyan | `#72D0D4` | Measured, linked, and ready |
| Calibration amber | `#E9A84A` | Conditional, estimated, or awaiting proof |
| Fault coral | `#F07C73` | Failed, unsafe, or blocked |
| Paper | `#EDF3F4` | Primary foreground and high-contrast surface |

Do not add more status colors. Shape, icon, text, and color must all carry status.

### Typography

- Use the macOS system face for native controls and window structure.
- Keep Familjen Grotesk for workbench titles and decision statements.
- Keep Atkinson Hyperlegible for dense explanatory text in the WebView.
- Keep IBM Plex Mono for hashes, IDs, measurements, provenance, and runtime state.

This preserves the existing identity while respecting the host platform.

### One coherent window

```text
┌──────────────────────┬───────────────────────────────────────────────────────┐
│ APTUS                │ Project: Parish Corpus Adapter      Runtime: MLX-LM  │
│                      ├───────────────────────────────────────────────────────┤
│ Projects             │ Facts   Compare   Compile   Validate   Run           │
│  • Current project   ├───────────────────────────────┬───────────────────────┤
│  • Recent            │                               │ FIT LEDGER            │
│                      │ Guided work area              │                       │
│ Runtimes             │                               │ Estimate              │
│ History              │ One decision at a time        │ Measured              │
│ Settings             │ Evidence beside each input    │ Remaining uncertainty │
│                      │                               │                       │
├──────────────────────┴───────────────────────────────┴───────────────────────┤
│ Status and diagnostics                         Primary next action          │
└───────────────────────────────────────────────────────────────────────────────┘
```

At compact widths, retain the existing inline Fit Ledger disclosure and refine
it into a focused drawer when more detail is needed. The stage control remains
inside the project. The global sidebar never repeats stage navigation.

### Signature interaction

The Fit Ledger should show scoped evidence milestones without turning analytic
segments into apparently measured components. Keep the component ledger visibly
estimated. Add separate markers for synthetic preflight, exact-model pilot, and
full-run peak. Each marker names its scope and identity binding. Unresolved
uncertainty remains amber and names the missing gate. One brief transition can
mark a new measured milestone. Respect reduced-motion settings and provide the
same change in text.

### Stage redesign

**Facts:** Begin with model, data, and goal. Inspect and profile after explicit
selection. Present missing attestations as a short review queue. Move detailed
architecture and runtime controls into Advanced Facts.

**Compare:** Lead with the recommendation, its bounded meaning, and the reason.
Show close alternatives beneath it. Preserve all unsupported candidates in an
expandable evidence table. Let the main container choose table or cards.

**Compile:** Show exactly what Aptus will write, where it will write, and why the
bundle is reproducible. Keep no-clobber behavior visible.

**Validate:** Display a vertical evidence ladder. Each rung should show identity,
result, provenance, timestamp, and the next missing proof.

**Run:** Show the five required actions as one explicit sequence. Each action
should have preparation, live output, result, and recovery guidance. Never hide
confirmation or computational cost.

**History:** Group plans, bundles, validations, and jobs under a named project.
Let users compare revisions and reveal artifacts without reconstructing IDs.

### Design self-critique

The current aesthetic is attractive, but the native card stack can resemble a
generic dashboard. Reduce nested cards. Use contiguous evidence sections,
Mac lists, and clear separators. Keep the instrument metaphor in measurements
and provenance, not decorative gauges.

The React screen currently displays too many banners, headers, and persistent
controls before the main comparison. Remove repeated explanations after the
user has acknowledged them. Detail should remain available on demand.

## Delivery sequence

### Release 0: Truth and reliability

- Fix process-tree timeout semantics, then diagnose the intermittent assertion.
- Complete and archive the canonical MLX-LM acceptance run.
- Run the complete desktop build ten consecutive times.
- Correct product and security claims, then supersede stale review records.
- Publish a diagnostic bundle command for support cases.

### Release 1: One product surface

- Make the workbench the primary native detail surface.
- Replace the duplicate native Data, Plans, and Runs placeholders.
- Retain and test the existing latest-session restore.
- Fix comparison container behavior and sticky-action overlap.
- Add first-launch runtime discovery, validation, and environment readiness.

### Release 2: Versioned product foundations

- Define versioned job and project persistence with migrations and corruption handling.
- Define explicit API response models and a checked versioned OpenAPI artifact.
- Generate TypeScript transport types and verify Swift models against the schema.
- Extract public storage and environment-binding primitives with portable parity tests.
- Add the named project repository before exposing project-library UI.
- Add bundle-compilation smoke tests for source, installed wheel, and frozen sidecar.

### Release 3: Maintainable seams

- Extract generated runtimes with byte-parity tests.
- Split execution, validation, and API routers behind tested compatibility boundaries.
- Introduce the React workflow reducer with concurrency and identity tests.
- Centralize product and contract version verification.
- Split native and CSS files only after the accepted window design stops moving.
- Enforce focused file-size targets on new and migrated modules. Document narrow
  exceptions for deliberately portable files such as `plan_contract.py`.

### Release 4: Guided usefulness

- Replace expert-first Facts with guided intake and advanced review.
- Add dataset preview, schema detection, and row-level remediation.
- Add a model picker or recent-model list while preserving immutable revision pinning.
- Add project history, revision comparison, artifact reveal, and recovery guidance.
- Make the canonical acceptance run available as an in-product tutorial.

### Release 5: Public Mac delivery

- Add Developer ID signing, notarization, and stapling.
- Test clean installs on macOS 15 and macOS 26.
- Verify quarantine launch, sidecar startup, runtime selection, and app removal.
- Publish signed review artifacts and an explicit update policy.

## Product success gates

- A first-time technical user reaches a validated example bundle within five minutes.
- A returning user reopens a named project and sees its exact evidence chain.
- No main workflow at supported window sizes needs horizontal page scrolling.
- Every recommendation names its candidate set, objective, assumptions, and missing proof.
- Ten consecutive authoritative builds pass on the release host.
- The shutdown stress test completes 1,000 iterations without an early completion.
- A real MLX-LM acceptance run succeeds twice from a clean checkout.
- Public artifacts pass Developer ID, notarization, stapling, and quarantine checks.
- Keyboard-only, VoiceOver, 200 percent zoom, Increased Contrast, and both appearances pass the acceptance checklist.

## Verification performed for this review

- Inspected Python, React, Swift, tests, build scripts, active documentation, and release policy.
- Built the full dependency and relationship graph for code, documentation, and product contracts.
- Ran `desktop/macos/build.sh` twice.
- Observed one intermittent native shutdown-test failure on the first full run.
- Reran the affected test successfully.
- Completed a second full build successfully.
- Verified 276 Python tests, 46 React tests, and 68 native tests in the passing build.
- Launched the built app and inspected Home, Machine, Models, Facts, and Compare.
- Loaded the bundled example and exercised the Compare transition.
- Confirmed creation of `desktop/macos/dist/Aptus.app` and `Aptus-macOS-arm64.dmg`.

## Approval boundary

This review intentionally changes no product behavior. The requested
architecture-review process requires approval before implementation begins.

Please review the findings and choose the first delivery slice. The recommended
slice is Release 0, followed by the unified product surface from Release 1.
