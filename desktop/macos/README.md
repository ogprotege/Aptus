# Aptus for Mac

Aptus for Mac is the native host for the existing Aptus workbench. AppKit owns
the application lifecycle, WebKit presentation, private backend session, native
file dialogs, and Finder integration. Python remains the source of truth for
planning, compilation, validation, and job rules. React remains the workbench.

## Current Mac scope

The Mac application profiles local datasets, inspects pinned model metadata,
compares declared fine-tuning candidates, compiles portable bundles, runs static
validation, and exports the result for a CUDA target host. Apple Silicon
inventory remains fail-closed for local CUDA execution. The desktop shell does
not claim MPS or MLX training support.

## Requirements

- macOS 13 or newer on Apple Silicon
- Xcode and the macOS SDK
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
desktop/macos/dist/Aptus-macOS-arm64.dmg
```

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

## Native bridge

The host injects this complete object before the workbench document loads:

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
