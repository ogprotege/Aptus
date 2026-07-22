# Validation States

Validation states form an evidence ladder. A report can also be `invalid` or
`unsupported`.

| State | Meaning | Does not prove |
|---|---|---|
| `contract-pass` | Required files and top-level contracts parse | Generated source safety or dependencies |
| `static-pass` | Identity, manifest, hashes, source structure, paths, configuration, and direct-pin rules pass | Imports, model load, CUDA fit |
| `dependency-pass` | The exact direct pins are installed and the environment binding is recorded | Model or data compatibility |
| `model-data-pass` | Pinned model and tokenizer resolve, loaded structural facts and target modules match, the selected method passes its plan-bound trainable-scope and LoRA-pair census, and every canonical row transforms | CUDA memory behavior or a persistent model-data census artifact |
| `measured-preflight-pass` | Selected method initializes and executes the synthetic CUDA check with bound peak metrics and a valid method-scope census | Real-model and real-data checkpoint behavior |
| `pilot-pass` | Two fresh bounded pilot phases pass with identical trainable censuses, current bindings, positive measured CUDA peaks, artifact contracts, and checkpoint continuation | Full-run completion or quality |
| `execution-approved` | Parent has admitted or observed a pending full run against a valid pilot | Successful training completion |
| `measured-run-pass` | Parent verified completed-run metrics, trainable census, full-dataset split evidence, and the structural export file tree | Task quality, safety, or deployment fitness |

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

Runtime states bind the relevant bundle manifest, plan ID, candidate ID, source
dataset digest, model revision, installed environment, hardware identity, and
prior artifacts. Measured-preflight metrics bind the selected method's census.
Pilot metrics bind both phase censuses and require them to be identical. The
model-data action enforces the same scope rules but does not persist a separate
census object in the validation report.

Full completion binds its run ID, metrics, ranks, census, exact optimizer
membership, and export manifest.
Its dataset-split evidence includes the exact strategy identifier, canonical
JSONL digest, assignment digest, row and unit counts, declared-group counts,
requested evaluation size, realized fraction, and row error. Declared groups
remain atomic, so grouped data can miss the requested fraction.

Changing a compiler-managed file invalidates the bundle. Changing environment or
hardware can invalidate runtime authorization even when a historical report
still says `pilot-pass`.

## Authorization display

UI and bootstrap responses can show cached status and capacity evidence. Deep
authorization occurs atomically when the train job is submitted. Admission is
the authoritative current check.
