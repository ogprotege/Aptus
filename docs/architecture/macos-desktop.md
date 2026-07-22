# macOS Desktop Host

> **Status:** Active | **Audience:** Contributors and release operators | **Authority:** Architecture | **Applies to:** Aptus 0.2 | **Owner:** Desktop | **Last reviewed:** 2026-07-22 | **Review by:** Every desktop packaging change

Aptus for Mac is a native Apple Silicon product. It does not implement a second
planner, compiler, or validation model. AppKit owns application lifecycle and
the main window. SwiftUI owns the Home, Machine, Models, Data, Plans, and Runs
shell. A bundled Python sidecar serves the FastAPI contracts and Aptus core.
The authenticated React workbench remains a contained transitional surface.

```text
AppKit application and window lifecycle
              |
              v
SwiftUI navigation shell
              |
              v
contained WKWebView workbench
              |
              v
authenticated ephemeral loopback
              |
              v
FastAPI, runtime dispatch, planner, compiler, validator, and jobs
```

## Ownership

The native host owns startup, failure recovery, shutdown, application paths,
private session creation, the six native destinations, file and folder panels,
Finder actions, and navigation policy. React owns the complete five-stage
workbench workflow. Python remains authoritative for facts, feasibility,
runtime selection, compilation, validation, jobs, leases, and evidence.

The Machine view reports the current chip, macOS version, processor capacity,
physical memory, Metal device, and Metal working-set advisory. The sidecar's
platform contract also reports an optional Metal GPU core count. Neither path
uses a chip-name allowlist. The Models view separates training runtime
configuration from LM Studio and oMLX inference availability.

Swift does not duplicate `TrainingPlan`, `CandidatePlan`, or
`ValidationReport`. The JavaScript bridge exposes only five members:

```text
platform
reportWorkbenchReady()
pickDataset()
pickOutputDirectory()
revealInFinder(path)
```

The browser build ignores an incomplete object. Ordinary browser operation does
not receive the bridge. Posted control requests time out after 30 seconds, and
modal file requests time out after five minutes. Native rejections use
`AptusDesktopError` with a stable `code`. Exact-origin malformed requests with
a valid identifier receive an `invalid_request` response.

The native `Choose MLX Python` panel is not part of the JavaScript bridge. It
shows hidden virtual-environment folders, accepts one executable, and sends the
exact path to `POST /api/v1/runtimes/configure` with the private session cookie.
The sidecar probes the runtime before persisting the canonical path. This avoids
relying on shell environment variables that Finder does not inherit.

## Process and session boundary

`aptus-desktop` binds `127.0.0.1` on an operating-system-selected port. It
writes a private readiness file containing only host, port, and version. The
native host creates a random 32-byte token, passes it through
`APTUS_DESKTOP_SESSION_TOKEN`, and installs it as an HttpOnly, SameSite Strict
cookie before loading the workbench.

When desktop authentication is enabled, protected API and schema routes require
the cookie. Health and static application assets remain public. Trusted-host
checking remains active. WebKit allows navigation only to the exact session
origin. User-activated external HTTP links open in the default browser. The
token never enters the readiness file, URL, JavaScript, state directory, or log.

Native API calls use an ephemeral URL session, the same exact origin and cookie,
bounded responses, and no redirects. They cannot change the endpoint origin.

## Storage

- Application state: `~/Library/Application Support/Aptus/state/`
- Backend log: `~/Library/Logs/Aptus/backend.log`
- Session readiness files: `~/Library/Caches/Aptus/sessions/`
- Runtime configuration: `~/Library/Application Support/Aptus/state/runtime-config.json`
- Compiled bundles and ZIP files: user-selected locations

State and session directories use user-only permissions. The app removes its
ephemeral session directory during normal shutdown. At startup, a backend log
at or above 2 MiB is compacted to its latest 2 MiB. Two mode-0600 archives are
retained, which bounds historical logs to 4 MiB plus the active session log.

## Apple runtime boundary

MLX-LM is a separate Apple training runtime. Its current compiler supports
single-device LoRA and QLoRA bundles. The Mac can run dependency and model-data
validation, a bounded measured preflight, an uninterrupted exact-model pilot,
and explicitly confirmed full-duration adapter training. All MLX-LM candidates
remain conditional until their exact gates pass.

The MLX pilot starts from the pinned base and proves at least two optimizer
updates, finite losses, exact target coverage, positive memory and adapter
delta, live headroom, and immutable artifacts. A fresh child loads the pinned
base plus adapter and generates one to four tokens. That reload is inference
evidence only. MLX crash resume is unsupported, and periodic saves are weight
snapshots rather than resumable checkpoints.

PyTorch MPS can be discovered and configured as an external runtime, but it has
no current compiler. The `mps` compute-backend value does not imply a PyTorch
MPS bundle. Shared unified memory is never presented as dedicated VRAM.
Device free VRAM remains unknown. MLX planning uses current free host RAM as its
live unified-memory headroom cap when the host can measure it. Pilot, reload,
and full-run code recheck live admission, while train submission compares
current headroom with measured pilot pressure plus reserve.

CUDA remains an NVIDIA target-host path. The Mac UI shows the ordered handoff
commands for CUDA bundles and does not reinterpret manual CUDA facts as local
capability.

LM Studio and oMLX are bounded loopback inference integrations. They can list
models and generate text. They cannot satisfy a training runtime or validation
gate.

## Platform design boundary

macOS 26 is the primary visual and release design. The code uses guarded system
effects and semantic materials. The deployment target is macOS 15, whose
fallback keeps the same navigation, typography, adaptive colors, accessibility,
and product boundaries without requiring macOS 26 effects.

## Packaging

`desktop/macos/build.sh` runs the product suite in isolated Python 3.12, rebuilds
and tests the web assets, creates a relocatable arm64 Python sidecar with
PyInstaller, generates the Xcode project, and runs native tests. It assembles
the app, creates the icon set, signs nested code and the outer bundle,
and emits a DMG. Local builds are ad-hoc signed. Public distribution still
requires Developer ID signing, notarization, and clean-machine verification.

## Related documentation

- [System architecture](system.md)
- [Security boundaries](security-boundaries.md)
- [Workbench development](../contributing/workbench.md)
- [Install Aptus](../getting-started/install.md)
