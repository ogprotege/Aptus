# Aptus for Mac

Aptus for Mac is a native Apple Silicon application shell. AppKit owns the
application lifecycle, private backend session, shutdown, native file dialogs,
and Finder integration. SwiftUI owns Home, Workbench, Machine, and Models. The
authenticated React workbench is inline in Workbench and owns the single Facts,
Compare, Compile, Validate, and Run workflow with project history.
Python remains the source of truth for planning, compilation, validation, and
job rules.

## Current Mac scope

The Mac application detects the actual chip, processor capacity, physical
memory, default Metal device, and Metal working-set advisory. The Machine view
also requests a nonblocking authenticated platform snapshot for current
available headroom and the system-wide free-memory percentage. It keeps the
8 GiB minimum MLX planning reserve separate from measured use and reports a
pilot peak as `Not measured` until a pilot produces evidence. It reports MLX
and Metal Performance Shaders as native Apple Silicon paths that still require
runtime, checkpoint, and workload validation. CUDA remains an external NVIDIA
target. A measured Metal GPU core count appears in technical details when the
platform probe supplies one. Aptus does not infer a Neural Engine core count or
turn installed memory or a detected runtime into a model-fit guarantee.

The Models view loads the authenticated runtime inventory without delaying the
native window. Its `selected` mapping restores a persisted MLX-LM interpreter
after relaunch. The native state distinguishes no selection, a measured
exact-pin-compatible selection, a persisted but unavailable or incompatible interpreter, an invalid
persisted path, and an inventory request that could not be verified.

The same inventory drives a read-only MLX environment doctor. It shows every
likely interpreter's path, discovery source, Python version, MLX-LM import
result, and exact-pin compatibility result. Only compatible rows offer **Use
this Python**. Selection still calls the backend's pinned runtime validator.
When none pass, the view
shows commands for an external environment with `mlx==0.31.2` and
`mlx-lm==0.31.3`. It installs nothing.

macOS 26 receives the current system glass treatment through guarded
availability checks. macOS 15 uses a semantic material fallback. Both paths use
system typography, adaptive colors, Reduce Transparency behavior, native
sidebar navigation, and one primary action per detail view.

## Requirements

- macOS 15 or newer on Apple Silicon
- Current Xcode and its installed macOS SDK
- XcodeGen
- Node.js and npm for the tested workbench build
- `uv` and Python 3.12 for the isolated sidecar build

## Build the distributable application

From the repository root:

```bash
desktop/macos/build.sh
```

The command runs the isolated Python 3.12, web, and native test gates. It builds
the authenticated Python sidecar and application, then renders the repository
SVG into an ICNS file,
signs the nested binary and application, and verifies the signature. It then
launches the packaged app and requires a visible window, a ready sidecar, a
successfully loaded authenticated workbench, and clean session teardown before
it creates the artifacts below. The workbench gate requires both the finished
same-origin document and a versioned React-ready handshake emitted only after
the authenticated bootstrap request succeeds.

```text
desktop/macos/dist/Aptus.app
desktop/macos/dist/Aptus.app.zip
desktop/macos/dist/Aptus-macOS-arm64.dmg
desktop/macos/dist/SHA256SUMS
desktop/macos/dist/COMMIT
```

The `Aptus desktop artifacts` GitHub Actions workflow performs this complete
build on the native arm64 `macos-26` runner for every pull request and push to
`main`. It uploads the DMG, a permissions-preserving ZIP of `Aptus.app`, and
`SHA256SUMS` plus a `COMMIT` source marker in an
`aptus-macos-arm64-<commit>` workflow artifact retained for 30 days. This keeps
generated binaries out of Git history without leaving the desktop deliverable
on one developer's machine.

Workflow artifacts use the default ad-hoc signature and are intended for
review and testing. Developer ID signing and notarization remain required for
a public release.

The default build creates an isolated Python 3.12 environment under the build
directory, installs `requirements-build.lock`, then installs the local Aptus
project with `--no-deps --no-build-isolation`. Debug builds may set
`APTUS_PYINSTALLER_PYTHON` to use an existing Python 3.12 environment instead.
Release builds reject that override and reject `--skip-tests` or `--skip-web`.
Set `APTUS_CODESIGN_IDENTITY` to a Developer ID Application identity for a
signed distribution build. The same identity is passed into PyInstaller so its
embedded extension binaries share the hardened-runtime signing identity. The
default ad-hoc signature is suitable only for local development. Developer ID
signing and notarization still require the corresponding local Apple credentials.

Set `APTUS_REQUIRE_CLEAN_CHECKOUT=1` to reject a dirty release checkout. Set
`APTUS_NOTARY_PROFILE` to a stored `notarytool` keychain profile to submit,
staple, validate, and assess the app and DMG. Set
`APTUS_REQUIRE_NOTARIZATION=1` to require both the Developer ID identity and
profile. Without those values the artifacts remain ad-hoc signed review builds.

After committing the exact release candidate, run the ten-build stability gate:

```bash
tools/repeat_desktop_release_gate.zsh
```

It refuses a dirty checkout. Each complete build runs with
`APTUS_REQUIRE_CLEAN_CHECKOUT=1`, captures its log, verifies the app signature
and DMG, and records the app ZIP and DMG hashes. A pass writes
`RELEASE-GATE.tsv` and `release-gate-logs.zip` into `desktop/macos/dist/` and
adds their hashes to `SHA256SUMS`. A positive argument changes the count for
diagnosis. Only the default ten consecutive runs satisfy this release gate.

The desktop build lock is intentionally separate from generated training-bundle
dependencies. Its exact input pins live in `requirements-build.in`, while the
generated lock records SHA-256 hashes for every accepted wheel. Refresh it with:

```bash
uv pip compile \
  --python-version 3.12 \
  --python-platform aarch64-apple-darwin \
  --generate-hashes \
  --only-binary :all: \
  --no-annotate \
  --no-sources \
  --custom-compile-command 'uv pip compile desktop/macos/requirements-build.in --python-version 3.12 --python-platform aarch64-apple-darwin --generate-hashes --only-binary :all: --no-annotate --no-sources --output-file desktop/macos/requirements-build.lock' \
  --output-file desktop/macos/requirements-build.lock \
  desktop/macos/requirements-build.in
```

Then run the complete build and review both the pins and hashes before
committing. The build requires hashes and accepts binary wheels only.

For a native-only development build that uses the repository virtual
environment:

```bash
desktop/macos/build.sh --skip-backend --skip-dmg --skip-web --debug
```

The Debug application resolves `.venv/bin/python` from the development
repository root and runs the current `src/aptus/desktop.py` in isolated mode.

## Runtime boundary

The native controller creates a random 32-byte session token and passes it only
through `APTUS_DESKTOP_SESSION_TOKEN`. The Python entrypoint pre-binds an
ephemeral loopback port and writes a mode-0600 readiness file. WebKit receives
an HttpOnly, SameSite=Strict cookie before its first request. Navigation is
restricted to the exact loopback origin.

Persistent paths are:

```text
~/Library/Application Support/Aptus/state/
~/Library/Logs/Aptus/backend.log
```

On startup, a backend log at or above 2 MiB is reduced to its most recent
2 MiB and rotated. Aptus retains two private archives, `backend.log.1` and
`backend.log.2`, so historical log retention is bounded to 4 MiB. The active
log can grow during the current session and is bounded on the next launch.

Readiness files live under `~/Library/Caches/Aptus/sessions/` and are removed
when the native host stops.

Quit and restart complete only after typed process-tree shutdown success. The
controller binds observations to PID plus process-start identity, expands late
descendants, ignores zombies, and rejects PID reuse. If forced termination
still leaves a survivor, it retains the process, paths, token, and session
directory. It blocks a replacement backend and refuses application termination
until a later explicit stop retry succeeds. The backend log records survivor
state and signal attempts.

The authenticated loopback API exposes three separate Apple integration
contracts for native follow-on work:

```text
GET /api/v1/platform
GET /api/v1/runtimes
GET /api/v1/inference/services
```

`platform` contains Apple host facts. `runtimes` contains external MLX-LM,
PyTorch MPS, and CUDA interpreter probes plus the persisted `selected` mapping.
The native shell compares the selected MLX-LM path with the measured exact-pin
compatible set before describing it as configured. Import availability alone is
insufficient. `inference/services` contains
inference-only LM Studio and oMLX probes. Native callers must keep these
contracts separate and use the same exact-origin session boundary as WebKit.

The Models view also uses:

```text
POST /api/v1/runtimes/configure
```

Its native MLX Python panel exposes hidden virtual-environment folders, sends
the exact executable path through the authenticated loopback session, and lets
the backend validate and persist that choice. The action reads `Choose`,
`Change`, or `Replace` according to the hydrated state. A failed replacement
does not erase the previously persisted state. This is needed because
applications launched from Finder do not inherit shell environment variables
such as `APTUS_MLX_PYTHON`.

## Native bridge

The contained workbench host injects this complete object before its document
loads:

```ts
window.aptusDesktop = {
  platform: "macos",
  reportWorkbenchReady(): Promise<void>,
  pickDataset(): Promise<string | null>,
  pickOutputDirectory(): Promise<string | null>,
  revealInFinder(path: string): Promise<void>,
};
```

The browser application feature-detects the whole object. Browser-only Aptus
sessions continue to use typed paths and do not receive partial native behavior.
Posted native control requests expire after 30 seconds. Dataset and output
pickers allow five minutes because they wait for a user decision. Native
rejections expose a stable error code through `AptusDesktopError.code`,
including `invalid_request` for malformed messages that carry a valid request
identifier.
