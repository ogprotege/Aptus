# System Architecture

> **Status:** Active | **Authority:** Normative architecture overview | **Applies to:** Aptus 0.2 | **Audience:** Contributors, operators, and integrators | **Last reviewed:** 2026-07-22 | **Review by:** 2027-01-22 or when a system boundary changes

Aptus separates facts, planning, compilation, validation, execution, and
completion evidence. Each boundary has a distinct contract.

```mermaid
flowchart LR
  A["Explicit facts"] --> B["Profiler and planner"]
  B --> C["Identity-bound runtime contract"]
  C --> D["Compiler dispatch"]
  D --> E["Portable bundle"]
  E --> F["Dependency"]
  F --> G["Model-data"]
  G --> H["Runtime-specific measured preflight"]
  H --> I{"Training runtime"}
  I -->|"CUDA"| J["Two-phase pilot"]
  I -->|"MLX-LM"| O["Uninterrupted adapter pilot"]
  J --> K["Deep train admission"]
  O --> K
  K --> L["Unique full run"]
  L --> M["Parent verification"]
  M --> N["Measured-run-pass"]
```

## 1. Fact intake

The API, CLI, and workbench produce the same model, dataset, hardware, and
target contracts. Facts retain provenance. Dataset profiling reads local data.
Optional model inspection retrieves bounded provider metadata. Optional hardware
inspection measures the local server host.

Apple platform inspection is a separate contract. It reports operating system,
chip, CPU, unified memory, current memory and swap pressure, Metal guidance, and
an optional Metal GPU core count. It reports runtime capabilities without a chip
allowlist. Training-runtime inventory probes exact Python executables. LM Studio
and oMLX discovery remains inference-only.

The user remains responsible for model and data rights. Provider fields do not
become permission facts automatically.

## 2. Planning

The planner reads only the registry's four selectable `gated-executable`
methods, enumerates the versioned 12-row placement matrix, applies support rules,
calculates transparent point and upper memory estimates, and ranks viable
`feasible` and `conditional` candidates, with feasible candidates first. Plan
and candidate IDs derive from canonical semantic payloads. Changing a bound fact
changes identity. Experimental, research-only, and documentation-only methods
cannot enter enumeration.

Each candidate carries an `aptus.runtime-contract.v1` record. The record binds
the compute backend, training runtime, compiler, estimator, evidence
requirement, and export kind. CUDA uses the Transformers and PEFT contract.
MLX-LM uses a separate unified-memory estimator and supports only
single-device LoRA and QLoRA. PyTorch MPS is a known runtime without a compiler.
For `mps` planning, device free VRAM remains unknown. The MLX estimator uses
current free host RAM as the live unified-memory headroom cap when available.

Planning is analytic. It does not import the selected training stack or
allocate accelerator memory.

## 3. Compilation

The runtime-bound compiler writes to a temporary sibling directory, validates
the result, and publishes it atomically. It refuses a non-empty destination.
The bundle contains the full canonical training rows, a bounded pilot set, the
portable plan, evidence, direct pins, generated programs, configuration,
reports, and a hash manifest. Archive creation is deterministic and no-clobber.

CUDA bundles contain the pinned Torch, Transformers, PEFT, Accelerate, and
method-specific paths. MLX-LM bundles contain pinned MLX and MLX-LM dependencies,
MLX data splits, an MLX adapter configuration, and runtime-neutral metrics.
MLX-LM QLoRA requires a model revision with explicit four-bit MLX quantization
metadata. It never imports bitsandbytes.

## 4. Validation

Validation is monotonic in required evidence:

1. `contract`
2. `static`
3. `dependency`
4. `model-data`
5. `measured-preflight`
6. `pilot`

Each runtime level binds its output to the bundle, plan, candidate, dataset,
model revision, environment, hardware, and prior artifacts as applicable. A
higher state does not turn estimates into quality measurements.

CUDA model-data prepares the selected method and enforces its trainable scope.
Its measured preflight persists a synthetic-path census. Both real-model pilot
phases must carry identical census records before the pilot can pass.

MLX-LM model-data loads the pinned model revision and tokenizes every bound row.
Its measured preflight runs a bounded real adapter smoke, writes an MLX adapter,
and records completed optimizer work, exact target binding, positive adapter
delta, and a positive measured peak in `aptus.runtime-metrics.v1`. Its pilot
runs the exact model and data from the pinned base without interruption for at
least two optimizer updates. A fresh child then loads the emitted adapter and
generates one to four tokens. This proves adapter reload, not training resume.

## 5. Execution

The local `JobService` persists jobs and logs under a selected state root. It
allows one Aptus accelerator action at a time for the same user across state roots by
using a host-global lease. Portable `validate.py`, `preflight.py`, and `run.py`
use the same lease contract. Unrelated CUDA or Metal programs do not participate.

Dependency, model-data, preflight, pilot, and train are separate ordered job
submissions. The service blocks forward skips until the preceding report state
passes. An operator may rerun an earlier passed action. Each higher validation
action cumulatively rechecks its lower levels inside that job as
defense-in-depth. Runtime validation through the API must use the job endpoint
so it remains cancellable.

The job service resolves MLX-LM and PyTorch MPS against an exact external Python
interpreter. A configured path is probed before it is persisted and before it
can become the runtime for a job. The desktop sidecar is not assumed to contain
the user's training stack.

## 6. Full-run admission and completion

Train submission holds the global lease and record locks while it deeply
revalidates pilot and capacity evidence. It then assigns a unique job and run ID.
The training child writes pending evidence to that run directory.

The full trainer computes a deterministic group-aware split over the complete
canonical JSONL. It binds canonical and assignment digests, detects mutation,
requires distributed agreement, and records requested and realized evaluation
sizes. Its grouped subset solver reaches the target when the atomic group sizes
make that possible. It also recomputes the method-scope census before optimizer
construction, binds one LoRA A/B pair to each inspected target instance, and
requires exact optimizer membership.

After child success, the parent enters `verifying`. It checks metrics, split and
census evidence, bindings, rank records, final-export structure, paths, sizes,
and hashes. It persists the attestation and promotes the report to
`measured-run-pass` in an idempotent transaction. Child exit alone cannot
declare completion.

For MLX-LM, train admission re-verifies the owned pilot and requires current
available unified memory above its measured peak plus reserve. A confirmed full
run starts again from the pinned base and derives its uninterrupted duration
from the compiled train rows, micro-batch, accumulation, and maximum epochs. The
parent verifies completed updates, finite train and validation losses, exact
target binding, positive memory and adapter delta, immutable artifacts,
fresh-process bounded generation, and `aptus.mlx-final-export.v1` before
promotion. MLX weight snapshots do not support resume.

For direct portable execution on POSIX, `run.py` performs the same parent role,
holds the shared lease through completion promotion, and recovers complete
pending evidence before starting another run. Direct portable child execution
is fail-closed on Windows in v0.2. Use the managed service there.

## Interfaces

- Aptus for Mac: AppKit lifecycle, SwiftUI Home, Machine, Models, Data, Plans,
  and Runs shell, private backend session, and a contained transitional WebKit
  workbench. macOS 26 is the primary design and macOS 15 is the fallback.
- CLI: local scripting and operator use.
- FastAPI: same-origin local workbench and programmatic control.
- React workbench: five-stage planning and execution flow.
- Portable bundle: host-independent artifacts with local runtime requirements.

LM Studio and oMLX are loopback-only inference adapters. They do not enter the
training graph. Cloud provider adapters, evaluation targets, exporter plugins,
and MCP adapters are future seams.

## Related documentation

- [Code map](code-map.md)
- [Data and identity flow](data-and-identity-flow.md)
- [Artifact compiler](artifact-compiler.md)
- [Execution orchestrator](execution-orchestrator.md)
- [Security boundaries](security-boundaries.md)
- [macOS desktop host](macos-desktop.md)
