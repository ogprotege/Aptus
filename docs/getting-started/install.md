# Install Aptus

> **Status:** Active | **Authority:** Operational installation guide | **Applies to:** Aptus 0.2 | **Audience:** Users and contributors | **Last reviewed:** 2026-08-06 | **Review by:** 2026-10-27 or when packaging changes

## Requirements

- Python 3.11 or newer. Package metadata accepts newer interpreters; CI
  currently tests Python 3.11 and 3.12.
- Node.js only when rebuilding the React application.
- An Apple Silicon Mac for MLX-LM dependency, model-data, and measured-preflight
  actions, or a CUDA host for the complete CUDA evidence ladder.
- Enough local disk for the bundle, model cache, pilot artifacts, checkpoints,
  and final export.

The planner, compiler, contract checks, static checks, API, and workbench can run
without CUDA. That does not establish that a training candidate works on the
intended target host.

## Build Aptus for Mac

The native app requires Apple Silicon and macOS 15 or newer. macOS 26 is the
primary design and release environment. macOS 15 uses the tested fallback.
Install current Xcode, XcodeGen, Node.js, `uv`, and a `uv` Python 3.12 runtime.
From the repository root, run:

```bash
desktop/macos/build.sh
```

The build verifies the React application, packages Aptus and its server into a
relocatable arm64 sidecar, compiles the AppKit host, assembles and ad-hoc signs
the app, and creates these local artifacts:

```text
desktop/macos/dist/Aptus.app
desktop/macos/dist/Aptus.app.zip
desktop/macos/dist/Aptus-macOS-arm64.dmg
desktop/macos/dist/SHA256SUMS
desktop/macos/dist/COMMIT
```

Launch the app with Finder or:

```bash
open desktop/macos/dist/Aptus.app
```

The default development artifact is ad-hoc signed but not notarized for public
distribution. The app uses an ephemeral loopback port and a random private
session cookie. It does not expose the desktop API to the ordinary browser.
Application state lives under `~/Library/Application Support/Aptus`, and the
backend log lives under `~/Library/Logs/Aptus`.

GitHub Actions uses a different ZIP name from the local build. It uploads
`Aptus-macOS-arm64.dmg`, `Aptus-macOS-arm64.zip`, `COMMIT`, and
`SHA256SUMS` in an artifact collection named
`aptus-macos-arm64-<commit-sha>`.

For public signing and notarization, configure a Developer ID Application
identity and a stored `notarytool` keychain profile:

```bash
APTUS_REQUIRE_CLEAN_CHECKOUT=1 \
APTUS_REQUIRE_NOTARIZATION=1 \
APTUS_CODESIGN_IDENTITY='Developer ID Application: Example (TEAMID)' \
APTUS_NOTARY_PROFILE='aptus-notary' \
desktop/macos/build.sh
```

The build submits and staples both the app and DMG, validates their tickets,
and runs Gatekeeper assessment. Do not claim public readiness unless those
commands succeed with real Apple credentials. One exact notarized arm64
identity is recorded in the
[2026-08-13 public Mac packet](../operations/evidence/2026-08-13-desktop-public-release/README.md).

The Mac app runs supported local MLX-LM gates and shows an explicit target-host
handoff for CUDA bundles. It does not run CUDA on macOS.

## Configure an MLX-LM training runtime

The packaged sidecar does not absorb an arbitrary training stack. Create a
separate environment and install the current MLX-LM pins:

```bash
python3 -m venv /path/to/aptus-mlx-env
/path/to/aptus-mlx-env/bin/python -m pip install --upgrade pip
/path/to/aptus-mlx-env/bin/python -m pip install 'mlx==0.31.2' 'mlx-lm==0.31.3'
```

Open **Models** in Aptus for Mac, choose **Choose MLX Python**, and select
`/path/to/aptus-mlx-env/bin/python`. Aptus probes that exact executable before
persisting its absolute command path. It does not resolve away the virtual-
environment symlink. Finder-launched applications do not inherit your
shell environment, so this selection is the authoritative desktop path.

The Models environment doctor lists likely interpreters with their Python
version, import-probe result, and failure reason. A passing **Use this Python**
action still goes through the backend's pinned validation. If none pass, the
doctor displays the same external-environment commands above. It does not
install or modify packages.

The environment makes MLX-LM dependency and runtime checks possible. It does
not itself authorize training. The exact bundle must pass model-data, measured
preflight, and uninterrupted pilot before full-duration adapter training can be
confirmed.

## Development install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[server,test]'
```

Confirm the CLI and server imports:

```bash
aptus --help
aptus serve --help
aptus doctor --help
aptus diagnostics --help
```

## Rebuild the web application

The Python package includes a built workbench in `src/aptus/_web`. From the
repository root, run this web-only iteration gate:

```bash
npm --prefix web ci
npm --prefix web test
npm --prefix web run typecheck
npm --prefix web run build
```

`npm --prefix web run build` writes directly to `src/aptus/_web` and clears the
previous packaged assets. Then build a clean wheel and run the installed-wheel
smoke test. This component gate does not replace the canonical
[repository-wide quality gate](../../CONTRIBUTING.md#required-repository-wide-checks).

For interactive Vite development or a focused Vitest watcher, use separate
terminals:

```bash
npm --prefix web run dev
npm --prefix web run test:watch
```

## Serve locally

```bash
aptus serve --host 127.0.0.1 --port 8787
```

Keep the service on loopback. Every launch prints a new authenticated workbench
URL and the same secret as an API bearer token. Open the printed URL, not a
manually typed root URL. The service exchanges its query token for an HttpOnly,
SameSite Strict cookie and immediately redirects to a clean URL. Only health and
static workbench assets are public. Protected API requests require that cookie
or `Authorization: Bearer TOKEN`.

The CLI disables Uvicorn access logs, but terminal output and the printed URL
still contain the secret. Do not paste or persist them. Non-loopback mode is
blocked unless explicitly enabled. Even then, Aptus serves plain HTTP, so use an
approved TLS and network boundary.

The native Mac host uses a separate exact-origin handoff. It installs its
per-launch cookie before WebKit's first request and never places the token in a
URL. See [security boundaries](../architecture/security-boundaries.md).

## Bundle environments

Each bundle contains `requirements.txt`, an exact set of direct pins selected by
method and training runtime. Create the isolated environment outside the bundle.
The manifest rejects unexpected files, including an in-bundle `.venv` directory.

```bash
python -m venv /path/to/aptus-bundle-env
source /path/to/aptus-bundle-env/bin/activate
python -m pip install -r /path/to/bundle/requirements.txt
cd /path/to/bundle
python validate.py --level dependency
```

The file is not a complete transitive lock. Dependency validation records the
installed environment for later binding checks. Portable `python validate.py`
does not require the Aptus package in that environment. It validates the
bundle's canonical frozen policy snapshot and reproduces its compatibility
decision, but it cannot determine whether that snapshot is current on an Aptus
host whose registry is absent. Installed `aptus validate` and managed
`aptus run` use the host registry and require replanning when the bundle's
coherent frozen policy is no longer current. Install Aptus and the bundle
requirements into the same external environment because jobs inherit the
interpreter that launched the CLI.

Do not repair a stale bundle by changing its schema, snapshot, or digest.
Installed-host validation reports malformed, non-object, or resource-hostile
plan, manifest, trainer, and snapshot documents as controlled invalid input.
Package-free validation covers the plan, manifest, and snapshot boundaries; the
trainer configuration remains compiler-managed input to later generated
runtime entrypoints. Do not edit it. Recompile the bundle from trusted source
facts instead.

## Evidence status

The current development Mac cannot provide CUDA runtime evidence. Repeat the
complete five-action sequence on each claimed CUDA configuration. For MLX-LM,
run the complete five-action sequence on the target Apple Silicon host. Its
pilot and full run are uninterrupted adapter-training actions. A fresh adapter
reload verifies bounded generation, but crash resume remains unsupported.

## Related documentation

- [Choose your path](choose-your-path.md)
- [First planning-only run](first-plan.md)
- [Security boundaries](../architecture/security-boundaries.md)
- [Bundle manifest](../reference/bundle-manifest.md)
