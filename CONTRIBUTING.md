# Contributing

## Scope

Aptus changes must preserve evidence boundaries. Planner output is a reasoned
comparison over a declared candidate catalog. Runtime validation is separate
evidence. Neither establishes model quality without an evaluation contract.

## Set up

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[server,test]'
cd web && npm ci
```

## Required checks

```bash
.venv/bin/ruff format --check src tests
.venv/bin/ruff check src tests
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m unittest discover -s tests -t . -v
PYTHONPATH=src .venv/bin/python -m compileall -q src tests
git diff --check
cd web && npm test && npm run typecheck && npm run build
```

Run the wheel build and installed-wheel smoke test for packaging changes. Run
real CUDA pilots for changes that alter generation, dependencies, precision,
quantization, distribution, memory estimates, checkpointing, or export.

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
new documents rather than rewriting historical audit records.

## Pull requests

Describe the changed contract, the tests run, the target hardware used, and any
evidence still missing. Include generated-bundle diffs when compiler output
changes. Never include private datasets, tokens, caches, checkpoints, or model
weights.
