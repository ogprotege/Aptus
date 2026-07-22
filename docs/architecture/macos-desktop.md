# macOS Desktop Host

> **Status:** Active | **Audience:** Contributors and release operators | **Authority:** Architecture | **Applies to:** Aptus 0.2 | **Owner:** Desktop | **Last reviewed:** 2026-07-22 | **Review by:** Every desktop packaging change

Aptus for Mac is a native host for the existing product. It does not implement
a second planner, compiler, or validation model. AppKit owns the window and
application lifecycle. WebKit renders the tested React workbench. A bundled
Python sidecar serves the same FastAPI contracts and Aptus core.

```text
AppKit window and startup state
              |
              v
WKWebView and native path bridge
              |
              v
authenticated ephemeral loopback
              |
              v
FastAPI, planner, compiler, and validator
```

## Ownership

The native host owns startup, failure recovery, shutdown, application paths,
private session creation, file and folder panels, Finder actions, and navigation
policy. React owns the five workflow stages and user-facing product state.
Python remains authoritative for facts, feasibility, compilation, validation,
jobs, leases, and evidence.

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

## Process and session boundary

`aptus-desktop` binds `127.0.0.1` on an operating-system-selected port. It
writes a private readiness file containing only host, port, and version. The
native host creates a random 32-byte token, passes it through
`APTUS_DESKTOP_SESSION_TOKEN`, and installs it as an HttpOnly, SameSite Strict
cookie before loading the workbench.

When desktop authentication is enabled, every route requires the cookie.
Trusted-host checking remains active. WebKit allows navigation only to the exact
session origin. User-activated external HTTP links open in the default browser.
The token never enters the readiness file, URL, JavaScript, state directory, or
log.

## Storage

- Application state: `~/Library/Application Support/Aptus/state/`
- Backend log: `~/Library/Logs/Aptus/backend.log`
- Session readiness files: `~/Library/Caches/Aptus/sessions/`
- Compiled bundles and ZIP files: user-selected locations

State and session directories use user-only permissions. The app removes its
ephemeral session directory during normal shutdown. At startup, a backend log
at or above 2 MiB is compacted to its latest 2 MiB. Two mode-0600 archives are
retained, which bounds historical logs to 4 MiB plus the active session log.

## macOS execution boundary

The current compiler executes only on CUDA. The Mac app can profile data,
inspect model metadata, compare manual target-host facts, compile a bundle, and
run static validation. It never submits dependency, model-data, preflight,
pilot, or training jobs locally. The Run stage instead displays the ordered
commands for the intended CUDA host. This is not only a hidden interface
control. The desktop sidecar rejects runtime validation and every job
submission with `desktop_execution_disabled`.

Apple Silicon hardware inspection remains useful inventory. It does not turn
shared unified memory into dedicated VRAM or make MPS or MLX executable.

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
