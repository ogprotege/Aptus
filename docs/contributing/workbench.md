# Workbench Development

> **Status:** Active | **Audience:** Frontend and API contributors | **Authority:** Operational | **Applies to:** Aptus 0.2 | **Owner:** Workbench | **Last reviewed:** 2026-08-06 | **Review by:** 2026-10-27

The React workbench is the complete transitional workflow inside the native Mac
product and a local browser interface over the same strict FastAPI contracts
used by the CLI. It must expose runtime identity, uncertainty, blocked actions,
current job state, and evidence boundaries without inventing another product
model.

## Source map

| Path | Responsibility |
|---|---|
| [`web/src/App.tsx`](../../web/src/App.tsx) | Application state, bootstrap restoration, stage transitions, polling, and active-job guards |
| [`docs/reference/openapi.v1.json`](../reference/openapi.v1.json) | Generated server contract from explicit Pydantic response models |
| [`web/src/generated/openapi.ts`](../../web/src/generated/openapi.ts) | Generated TypeScript schema and path types; not a complete SDK |
| [`web/src/api.ts`](../../web/src/api.ts) | Maintained API requests, request/receipt-correlated plan and typed no-feasible ingress, generated-type consumption, and error handling |
| [`web/src/types.ts`](../../web/src/types.ts) | Generated model-policy aliases plus maintained browser facts, plans, candidates, reports, jobs, and presentation types |
| [`web/src/stages/`](../../web/src/stages) | Facts, Compare, Compile, Validate, and Run screens |
| [`web/src/components/`](../../web/src/components) | Workflow rail, model-policy records, candidate comparison, fit ledger, validation gates, artifact tree, and run console |
| [`web/src/lib/modelPolicy.ts`](../../web/src/lib/modelPolicy.ts) | Strict v2 decision, path, receipt, candidate, binding, and validation-report decoders plus policy presentation |
| [`web/src/lib/modelInspection.ts`](../../web/src/lib/modelInspection.ts) | Provider and plan-derived fact application without browser policy reconstruction |
| [`web/src/demo.ts`](../../web/src/demo.ts) | Labeled non-executed example content |
| [`web/src/desktopBridge.ts`](../../web/src/desktopBridge.ts) | Complete native-bridge feature detection and browser fallback |
| [`web/src/styles.css`](../../web/src/styles.css) | Fonts, tokens, layout, status treatments, responsive rules, focus, and reduced motion |
| [`desktop/macos/`](../../desktop/macos) | AppKit lifecycle, SwiftUI shell, contained WebKit host, native API client, packaging, and Mac tests |

## Five-stage contract

1. Facts collects model, dataset, hardware, and target values with provenance.
2. Compare displays every candidate status and the selected recommendation.
3. Compile writes a new no-clobber bundle and archive.
4. Validate shows the evidence ladder and starts static validation.
5. Run exposes dependency, model-data, preflight, pilot, and confirmed training
   as separate ordered actions.

The UI can guide the sequence. It cannot bypass the API's strict schemas,
planner rules, manifest checks, job prerequisites, host-global lease, current
train admission, or parent completion verification.

## API ownership

The browser should obtain capability and readiness data from
`GET /api/v1/bootstrap`. `capabilities.methods` is the planner-selectable set.
`capabilities.method_catalog` is the wider runtime registry. Never populate the
preference control from the wider catalog.

Apple-specific data stays in separate contracts:

- `GET /api/v1/platform` reports the Apple host and runtime facts;
- `GET /api/v1/runtimes` reports exact Python interpreter probes;
- `POST /api/v1/runtimes/configure` validates and persists one interpreter;
- `GET /api/v1/inference/services` reports LM Studio and oMLX availability;
- inference model and generation requests never populate training-runtime
  state.

When adding an API field:

1. change the Pydantic model and endpoint;
2. add API success and rejection tests;
3. regenerate the OpenAPI JSON and TypeScript schema and path map;
4. update request construction or response normalization in `api.ts`;
5. update `types.ts` only when the maintained UI domain model changes;
6. update applicable Swift decoders and their contract check;
7. update restoration and stage state in `App.tsx`;
8. add component or stage tests;
9. update the API and UI documentation.

Run the contract workflow from the repository root:

```bash
uv run --isolated --python 3.12 --locked --extra server --extra test \
  python tools/generate_openapi.py
npm --prefix web run openapi:generate
uv run --isolated --python 3.12 --locked --extra server --extra test \
  python tools/generate_openapi.py --check
npm --prefix web run openapi:check
uv run --isolated --python 3.12 --locked --extra server --extra test \
  python tools/check_client_contracts.py
uv run --isolated --python 3.12 --locked --extra server --extra test \
  python tools/verify_versions.py
```

Generated TypeScript types provide compile-time alignment. They do not validate
untrusted responses at runtime. Keep the maintained normalizers fail closed on
missing, malformed, unknown-version, or misbound data.

For model policy, decode the exact object keys and supported schema versions for
the server-produced `aptus.model-compatibility.v2` decision, every nested path,
the optional `aptus.model-inspection-receipt.v1`, and each candidate's explicit
nullable `aptus.model-policy-binding.v1`. Cross-check decision IDs, subject
digests, source and receipt identities, path membership, and the candidate's
method, distribution, targets, and runtime contract. Do not add family-specific
policy predicates to React. The inspection receipt's v2 decision is the one
browser policy source; do not restore the retired flattened-compatibility
normalizer or create a second inspection projection.

The planning endpoint's HTTP 422 `no_feasible_plan` variant is a closed typed
response, not merely a candidate list. Its decision, source, nullable receipt,
required `model` subject, candidate links, and any non-null bindings must pass
the same chain validation as a successful plan. Correlate `model.model_id` and
`model.revision` with the submitted artifact even for an unreceipted
user-attested failure, then verify the expected policy source and receipt ID.
Decode every candidate's method, distribution, status, feasibility, rejection
reasons, target modules, runtime contract, decision link, and binding; require
all rows to be rejected. Preserve them only as a non-compilable partial
comparison.

In particular, every purported v5 plan response must carry
`model_policy_snapshot_sha256` as lowercase 64-character hexadecimal text. The
normalizer rejects a missing, non-string, uppercase, short, or non-hexadecimal
value. Keep HTTP `409 replan_required` as a distinct structured lifecycle result
for a coherent saved plan that is no longer current; do not collapse it into a
generic `invalid_request` message or repair the saved plan in the browser.

Preserve `null` for unknown resource values. Do not turn a missing free-memory
measurement into total memory.

## Safety invariants

Keep these behaviors visible and tested:

- model inspection cannot confirm training permission or choose a license;
- inferred model family remains distinct from provider-declared fields;
- hardware scanning names the service host;
- Apple shared unified memory is not labeled dedicated VRAM;
- Apple local scans select MLX-LM, preserve unknown free VRAM, use current free
  host RAM as live headroom when present, and apply the local 8 GiB minimum
  reserve;
- CUDA BF16 and bitsandbytes flags are not reused as MLX capabilities;
- PyTorch MPS remains non-executable until a compiler exists;
- LM Studio and oMLX remain inference-only;
- experimental and research-only methods remain nonselectable;
- conditional candidates retain their unresolved reasons;
- model-policy match, selected candidate path, and evidence readiness remain
  three separate records;
- exact path equality requires a non-null candidate binding, while truly unbound
  and rejected rows receive no synthesized policy ladder or validation action;
- a provider path-matched receipt requires provider-declared provenance and
  cannot be satisfied by inferred-only observations;
- the decoded recommendation structurally equals its complete listed candidate
  record, not merely a subset of execution fields;
- validation evidence applies only when its report binds the current plan ID,
  selected candidate ID, and immutable model revision, and that same exact tuple
  gates stage completion plus validation and run actions;
- evidence incomplete/complete remains distinct from the optional typed
  `authorization_status` values `current`, `deferred`, and `blocked`; a tuple
  with no non-null member means not checked;
- `current` pairs with `authorization_current: true` and no error, while
  `deferred` or `blocked` pairs with false and a non-empty diagnostic; partial or
  contradictory tuples fail closed;
- authorization state is never inferred from diagnostic prose, and a generic
  training-request failure surfaces its error without mutating the prior report;
- non-current authorization is not itself a stale-policy or replan result;
- the MoE topology rail explains routed activity and resident weights without
  reconstructing policy or reducing residency by active parameters;
- typed `no_feasible_plan` responses preserve and validate the complete policy
  chain before rejected candidates render;
- compile requires a new path and explains no-clobber behavior;
- v5 plan responses require an exact model-policy snapshot digest;
- `replan_required` preserves the old plan and directs the user to create and
  compile a new current plan;
- runtime actions cannot skip forward;
- current train admission is authoritative, not cached bootstrap text;
- a portable frozen-snapshot pass is not presented as current-host policy
  authorization;
- training requires explicit high-cost confirmation;
- `verifying` is shown before completion;
- full-run resume is not offered;
- export checks are labeled structural, not quality evidence;
- example mode is labeled as non-executed on every relevant stage;
- macOS desktop mode runs eligible MLX-LM actions locally, distinguishes its
  uninterrupted pilot from CUDA checkpoint continuation, and enables confirmed
  full-duration training only after current `pilot-pass` evidence;
- MLX status text calls periodic files weight snapshots, rejects resume, and
  describes fresh-process generation as adapter reload rather than continuation;
- macOS desktop mode never exposes local CUDA run controls, including when
  manual facts describe a different CUDA host.

## Active-job behavior

Facts that can conflict with an executing host, local hardware scanning,
compilation selection, and competing runtime submissions are blocked while a
managed job is active. Polling should remain cheap. Cancellation stays
available until the parent enters its non-cancellable completion commit.

Display both `state` and `phase`. A job can be `running` with phase
`verifying`. Do not map phase text into a false persisted state.

## Accessibility requirements

- Give every input a programmatic label and useful error association.
- Move focus to the selected stage heading.
- Announce ordinary status changes through a polite live region.
- Use an alert for blocking errors.
- Never use color as the only state signal.
- Keep every action and disclosure keyboard reachable.
- Preserve action order and evidence labels on narrow screens.
- Respect reduced-motion preferences.
- Maintain visible focus and readable contrast in the packaged build.

Test semantic behavior with Testing Library queries that reflect how a user
finds the control. Avoid tests that pass only because of internal component
structure.

## Build and test

From the repository root, run this workbench-only iteration gate:

```bash
npm --prefix web ci
npm --prefix web run openapi:check
npm --prefix web test
npm --prefix web run typecheck
npm --prefix web run build
```

`npm --prefix web run build` runs the type check, then writes directly to
`src/aptus/_web`. Vite clears the previous output. The Python package includes
that directory as package data. This component gate does not replace the
canonical [repository-wide quality
gate](../../CONTRIBUTING.md#required-repository-wide-checks).

After the build:

1. inspect the changed hashed assets and `index.html`;
2. run the Python API tests;
3. build a wheel;
4. install it outside the source tree;
5. fetch `/` and the referenced hashed asset through the packaged FastAPI app.

Do not commit a source change without the corresponding packaged assets when
the user-facing build changed.

For a full native package, return to the repository root and run:

```bash
desktop/macos/build.sh
```

This also verifies the injected bridge, authenticated sidecar startup, AppKit
and SwiftUI host, contained workbench, ad-hoc signature, and final app and DMG
layout.

## Local development

Run the API on loopback:

```bash
aptus serve --host 127.0.0.1 --port 8787
```

In a second terminal at the repository root, start Vite:

```bash
npm --prefix web run dev
```

For focused test-driven work, a third terminal can keep Vitest in watch mode:

```bash
npm --prefix web run test:watch
```

The command prints a per-launch authenticated workbench URL and bearer token.
Open the printed backend URL once. Its query handoff sets an HttpOnly, SameSite
Strict cookie and returns `303` without the token query. The static application
and health routes are public, but bootstrap and every other product API require
the cookie or bearer header. Uvicorn access logs are disabled by this command.

The Vite development configuration uses `127.0.0.1:4173` and proxies `/api` to
`http://127.0.0.1:8787` by default. `APTUS_API_ORIGIN` can select another local
API origin for the proxy. `VITE_API_BASE_URL` can set a browser API base at
build or development time. With the default same-host proxy, establish the
cookie through the printed backend URL before opening Vite.

Do not use development proxy settings as a remote deployment security model.
The per-launch token authenticates one local session. It does not add tenant
isolation, filesystem scoping, worker isolation, or TLS. Explicit non-loopback
serve mode sends the credential over plain HTTP and requires an approved TLS
and network boundary.

## Visual and interaction review

For any material UI change, inspect:

- empty, loading, success, warning, failure, cancelling, and verifying states;
- all five workflow stages;
- all six native Mac destinations;
- a conditional and unsupported candidate;
- method readiness on CUDA and MLX-LM;
- the macOS 26 appearance and macOS 15 fallback;
- exact MLX Python selection and failure states;
- LM Studio and oMLX inference-only labels;
- long paths, error text, and logs;
- keyboard-only operation;
- reduced motion;
- narrow and wide viewports;
- example-mode labels;
- the installed-wheel build, not only the Vite source server.

## Related documentation

- [UI and UX contract](../product/ui-ux.md)
- [API reference](../reference/api.md)
- [Current capabilities](../product/current-capabilities.md)
- [Security boundaries](../architecture/security-boundaries.md)
- [Code map](../architecture/code-map.md)
- [macOS desktop host](../architecture/macos-desktop.md)
