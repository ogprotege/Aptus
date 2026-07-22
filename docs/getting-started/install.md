# Install Aptus

> **Status:** Active | **Authority:** Operational installation guide | **Applies to:** Aptus 0.2 | **Audience:** Users and contributors | **Last reviewed:** 2026-07-22 | **Review by:** 2026-10-22 or when packaging changes

## Requirements

- Python 3.11 or newer.
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
desktop/macos/dist/Aptus-macOS-arm64.dmg
```

Launch the app with Finder or:

```bash
open desktop/macos/dist/Aptus.app
```

The development artifact is locally signed but not notarized for public
distribution. The app uses an ephemeral loopback port and a random private
session cookie. It does not expose the desktop API to the ordinary browser.
Application state lives under `~/Library/Application Support/Aptus`, and the
backend log lives under `~/Library/Logs/Aptus`.

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
persisting its canonical path. Finder-launched applications do not inherit your
shell environment, so this selection is the authoritative desktop path.

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
```

## Rebuild the web application

The Python package includes a built workbench in `src/aptus/_web`. To change it:

```bash
cd web
npm ci
npm test
npm run typecheck
npm run build
```

`npm run build` writes directly to `src/aptus/_web` and clears the previous
packaged assets. Then build a clean wheel and run the installed-wheel smoke test.

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
does not require the Aptus package in that environment. Managed `aptus run`
does: install Aptus and the bundle requirements into the same external
environment because jobs inherit the interpreter that launched the CLI.

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
