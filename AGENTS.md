# AGENTS.md

## Cursor Cloud specific instructions

### Scope on this Linux VM
Aptus has four surfaces (see the README "Architecture at a glance" table). Only two are runnable on a standard Linux cloud VM:

- Python application (`src/aptus/`): planner, compiler, validator, FastAPI API, and the `aptus` CLI.
- React workbench (`web/`): the Facts → Compare → Compile → Validate → Run UI (Vite dev server, Vitest, build).

The macOS desktop app (`desktop/macos/build.sh`) needs Xcode/Apple Silicon, and CUDA/MLX-LM *training runtime* execution needs GPU/Apple hardware. These are out of scope here and fail closed by design (e.g. `aptus hardware` reports `unavailable` with no accelerator). Planning, compiling, and validating are pure arithmetic and run fine without a GPU.

### Environment layout
- The Python virtualenv lives at `.venv/` (git-ignored). Use `.venv/bin/aptus`, `.venv/bin/python`, `.venv/bin/ruff`. There is no bare `python` on PATH — use `python3`.
- Web dependencies are installed under `web/node_modules/` via `npm --prefix web ci`.

### Lint / test / build / run commands
The canonical, full repository quality gate is in [CONTRIBUTING.md](CONTRIBUTING.md#required-repository-wide-checks) — run those exact commands from the repo root. Web scripts are in `web/package.json`; CLI subcommands are in the README command reference. Quick pointers:

- Python tests: `PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests -t .` (~1000 tests, ~90s).
- Python lint: `.venv/bin/ruff check src tests tools` and `.venv/bin/ruff format --check src/aptus tests/aptus`.
- Web: `npm --prefix web test` (Vitest), `npm --prefix web run typecheck`, `npm --prefix web run build`, `npm --prefix web run dev` (Vite dev server).

### Non-obvious gotchas
- `npm --prefix web run build` (and CI's build) emits the packaged workbench into `src/aptus/_web/` (not a `web/dist/`). `aptus serve` and the wheel serve those built assets, so build the web app before serving if you want fresh UI. This directory is git-ignored.
- `aptus serve --host 127.0.0.1 --port 8787` mints a **fresh session token on every launch** and prints it. To connect the browser workbench to the API, open the origin with the token as a **query parameter**: `http://127.0.0.1:8787/?aptus_session_token=<TOKEN>` (this sets an auth cookie and redirects). The URL `#token=...` *fragment* does NOT work because fragments are never sent to the server. The same token value is the API Bearer token for `curl -H "Authorization: Bearer <TOKEN>"`.
- In the workbench, use "Load labeled example" on the Facts stage to populate example facts, then open the "Compare" stage to see the ranked candidate table / Fit Ledger without needing a real model or GPU.
