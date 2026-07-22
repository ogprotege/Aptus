# Install Aptus

> **Status:** Active | **Authority:** Operational installation guide | **Applies to:** Aptus 0.2 | **Audience:** Users and contributors | **Last reviewed:** 2026-07-22 | **Review by:** 2026-10-22 or when packaging changes

## Requirements

- Python 3.11 or newer.
- Node.js only when rebuilding the React application.
- A CUDA host for measured preflight, pilot, and training actions.
- Enough local disk for the bundle, model cache, pilot artifacts, checkpoints,
  and final export.

The planner, compiler, contract checks, static checks, API, and workbench can run
without CUDA. That does not establish that a training candidate works on the
intended target host.

## Build Aptus for Mac

The native app requires macOS 13 or newer, Xcode, XcodeGen, Node.js, `uv`, and
an installed `uv` Python 3.12 runtime. From the repository root, run:

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

The Mac app does not run CUDA actions. It prepares and statically validates the
bundle, then shows the ordered commands to run on the intended CUDA host.

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

Keep the service on loopback. The local jobs API can launch processes and has no
built-in authentication. See [security boundaries](../architecture/security-boundaries.md).
This statement applies to `aptus serve`. The native Mac host adds a per-launch
cookie boundary around its private service.

## Bundle environments

Each bundle contains `requirements.txt`, an exact set of direct pins selected by
method. Create the isolated environment outside the bundle. The manifest rejects
unexpected files, including an in-bundle `.venv` directory.

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

## CUDA evidence status

The current development Mac cannot provide the required CUDA runtime evidence.
Before release, repeat installation and the full five-action sequence on each
claimed CUDA configuration.

## Related documentation

- [Choose your path](choose-your-path.md)
- [First planning-only run](first-plan.md)
- [Security boundaries](../architecture/security-boundaries.md)
- [Bundle manifest](../reference/bundle-manifest.md)
