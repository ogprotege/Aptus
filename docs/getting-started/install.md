# Install Aptus

## Requirements

- Python 3.11 or newer.
- Node.js only when rebuilding the React application.
- A CUDA host for measured preflight, pilot, and training actions.
- Enough local disk for the bundle, model cache, pilot artifacts, checkpoints,
  and final export.

The planner, compiler, contract checks, static checks, API, and workbench can run
without CUDA. That does not establish that a training candidate works on the
intended target host.

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

## Bundle environments

Each bundle contains `requirements.txt`, an exact set of direct pins selected by
method. Install those pins in an isolated environment:

```bash
cd /path/to/bundle
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python validate.py --level dependency
```

The file is not a complete transitive lock. Dependency validation records the
installed environment for later binding checks.

## CUDA evidence status

The current development Mac cannot provide the required CUDA runtime evidence.
Before release, repeat installation and the full five-action sequence on each
claimed CUDA configuration.
