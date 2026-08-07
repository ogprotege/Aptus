# Contributing

> **Status:** Active | **Authority:** Normative contribution policy | **Applies to:** Aptus 0.2 | **Audience:** Contributors | **Last reviewed:** 2026-08-06 | **Review by:** 2026-10-27 or when the quality gate changes

## Scope

Aptus changes must preserve evidence boundaries. Planner output is a reasoned
comparison over a declared candidate catalog. Runtime validation is separate
evidence. Neither establishes model quality without an evaluation contract.

## Set up

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[server,test]'
npm --prefix web ci
```

Package metadata accepts Python 3.11 or newer. The required CI matrix currently
tests Python 3.11 and 3.12; a newer interpreter allowed by metadata is not part
of that current test matrix.

The test extra pins Ruff exactly. The explicit `[tool.ruff.lint]` selection in
`pyproject.toml` defines Aptus's lint policy, so a Ruff release cannot silently
expand the required rule set. Upgrade the pin, lock, policy, and formatting in
one reviewed change.

## Required repository-wide checks

This is the canonical full repository quality gate. Run every command from the
repository root; component guides may show smaller, explicitly partial gates
for faster iteration.

```bash
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m unittest discover -s tests -t . -v
.venv/bin/ruff check src tests tools
.venv/bin/ruff format --check src/aptus tests/aptus
PYTHONPATH=src .venv/bin/python -m compileall -q src tests tools
.venv/bin/python tools/generate_openapi.py --check
.venv/bin/python tools/check_client_contracts.py
.venv/bin/python tools/verify_versions.py
npm --prefix web run openapi:check
npm --prefix web test
npm --prefix web run typecheck
npm --prefix web run build
git diff --check
```

When the HTTP contract changes, regenerate both derived contract artifacts from
the repository root before running those checks:

```bash
.venv/bin/python tools/generate_openapi.py
npm --prefix web run openapi:generate
```

`docs/reference/openapi.v1.json` is the generated server contract.
`web/src/generated/openapi.ts` is the generated TypeScript schema and path map.
React still uses a maintained request, normalization, and presentation layer in
`web/src/api.ts` and `web/src/types.ts`. Swift response decoders are also
maintained source. `tools/check_client_contracts.py` checks their covered
endpoint and runtime-inventory boundary against OpenAPI.

Run the wheel build and installed-wheel smoke test for packaging changes. Run a
real pilot on every affected target runtime for changes that alter generation,
dependencies, precision, quantization, distribution, memory estimates,
checkpointing or MLX weight snapshots, reload, or export. CUDA changes require
the two-phase checkpoint-continuation pilot. MLX changes require the exact
model-data pilot with at least two optimizer updates and a fresh-process adapter
reload that generates one to four tokens.

For macOS host, bridge, packaged workbench, or desktop-runtime changes, also run:

```bash
desktop/macos/build.sh
```

Keep the resulting `desktop/macos/dist/` artifacts out of commits. Record native
test, signature, authenticated launch, and clean-machine results in release
evidence.

## Design rules

- Keep user-attested, provider-declared, inferred, measured, and unknown facts
  distinct.
- Pin model revisions. Do not convert provider metadata into a user permission
  decision.
- Give unsupported combinations an explicit reason and abstain.
- Keep point estimates and upper envelopes separate from measured peaks.
- Update plan identity and bundle contracts when semantics change.
- Refuse non-empty compiler outputs. Do not overwrite an existing run.
- Keep runtime actions ordered and cancellable through managed jobs.
- Treat `requirements.txt` as exact direct constraints, not a transitive lock.
- Keep full-training resume disabled until its complete state contract exists.
  Reject every MLX resume argument and call periodic MLX artifacts weight
  snapshots, not resumable checkpoints.
- Do not weaken parent-owned completion verification.

## Claim rules

- Do not call an estimate a guarantee.
- Use “recommended within the enumerated candidate set” when needed. Do not call
  a plan universally optimal.
- Do not claim quality from a training loss or a structural export check.
- Do not claim release readiness without the evidence record in
  [`docs/operations/release-gates.md`](docs/operations/release-gates.md).

## Documentation

Update user, API, bundle, validation, and capability documentation in the same
change as behavior. Preserve archived legacy-audit evidence. Add new findings as
new documents rather than rewriting historical audit records. Never edit the
generated OpenAPI JSON or TypeScript map by hand.

## Pull requests

Describe the changed contract, the tests run, the target hardware used, and any
evidence still missing. Include generated-bundle diffs when compiler output
changes. Never include private datasets, tokens, caches, checkpoints, adapter
weight snapshots, or model weights.

## Related documentation

- [Contributor index](docs/contributing/index.md)
- [Code map](docs/architecture/code-map.md)
- [Documentation policy](docs/maintenance/documentation-policy.md)
- [Release gates](docs/operations/release-gates.md)
