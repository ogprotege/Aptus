# Validation States

Validation states form an evidence ladder. A report can also be `invalid` or
`unsupported`.

| State | Meaning | Does not prove |
|---|---|---|
| `contract-pass` | Required files and top-level contracts parse | Generated source safety or dependencies |
| `static-pass` | Identity, manifest, hashes, source structure, paths, configuration, and direct-pin rules pass | Imports, model load, CUDA fit |
| `dependency-pass` | The exact direct pins are installed and the environment binding is recorded | Model or data compatibility |
| `model-data-pass` | Pinned model and tokenizer resolve, loaded parameter count and plan-driving structural config fields match, target modules are checked, and every canonical row transforms | CUDA memory behavior |
| `measured-preflight-pass` | Selected method initializes and executes the synthetic CUDA check with bound peak metrics | Real-model and real-data checkpoint behavior |
| `pilot-pass` | Two fresh bounded pilot phases pass with current bindings, positive measured CUDA peaks, artifact contracts, and checkpoint continuation | Full-run completion or quality |
| `execution-approved` | Parent has admitted or observed a pending full run against a valid pilot | Successful training completion |
| `measured-run-pass` | Parent verified the completed run metrics and structural export file tree | Task quality, safety, or deployment fitness |

## Requested levels

The public validation levels are:

```text
contract
static
dependency
model-data
measured-preflight
pilot
```

`execution-approved` and `measured-run-pass` are lifecycle states, not standalone
validation levels.

## Bindings

Runtime states bind the relevant bundle manifest, plan ID, candidate ID, dataset
digest, model revision, installed environment, hardware identity, and prior
artifacts. Pilot also binds its metrics. Full completion binds its run ID,
metrics, ranks, and export manifest.

Changing a compiler-managed file invalidates the bundle. Changing environment or
hardware can invalidate runtime authorization even when a historical report
still says `pilot-pass`.

## Authorization display

UI and bootstrap responses can show cached status and capacity evidence. Deep
authorization occurs atomically when the train job is submitted. Admission is
the authoritative current check.
