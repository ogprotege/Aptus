<p align="center">
  <img src="desktop/macos/Resources/AptusMark.svg" width="96" alt="Aptus mark">
</p>

<h1 align="center">Aptus</h1>

<p align="center"><strong>Plan, compile, and measure fine-tuning work on Apple Silicon or CUDA.</strong></p>

<p align="center">
  Aptus turns explicit model, dataset, hardware, and runtime facts into a ranked plan,<br>
  a runtime-bound bundle, and an evidence ladder that refuses unsupported claims.
</p>

<p align="center">
  <a href="#start-on-macos"><strong>Build the engineering preview</strong></a> ·
  <a href="docs/getting-started/first-plan.md">Run the tutorial</a> ·
  <a href="docs/index.md">Read the documentation</a>
</p>

<p align="center"><sub>Engineering preview · Apple Silicon · Source build</sub></p>

<details>
<summary>Project status</summary>

**Status:** Engineering preview | **Applies to:** Aptus 0.2<br>
**Last reviewed:** 2026-07-22 | **Review by:** 2026-10-22 or when the support contract changes

Aptus has separate CUDA and MLX-LM compiler contracts. Apple Silicon LoRA and
QLoRA candidates remain conditional until their exact bundle passes measured
gates. The MLX-LM path now supports bounded measured preflight, an uninterrupted
exact-model pilot, and explicitly confirmed full-duration adapter training from
the pinned base model. It does not support crash resume. The Mac build is
locally signed and not notarized. No real CUDA or Apple Silicon pilot has
completed the release gates yet.

</details>

## One decision, with the reasons attached

| You provide | Aptus evaluates | You receive |
| --- | --- | --- |
| A pinned model, supervised dataset, target hardware, and training goal | Which supported methods appear feasible, which fail, and why | A decision record and a no-clobber training bundle with code, configuration, evidence, and hashes |

Aptus compares Full, LoRA, int8-LoRA, and QLoRA across the placements it can
represent. It preserves infeasible and unsupported candidates instead of hiding
them. A recommendation means highest-ranked within this bounded Aptus catalog.
It is not a quality prediction or a guarantee that unmeasured hardware will fit.

## The five-stage workbench

| Stage | What happens |
| --- | --- |
| **Facts** | Profile local data and record the exact model, hardware, and training target. |
| **Compare** | Apply hard feasibility rules, inspect memory ledgers, and explain the ranking. |
| **Compile** | Write a new bundle directory and deterministic ZIP without overwriting prior work. |
| **Validate** | Check contracts, identities, generated source, paths, hashes, and direct dependency pins. |
| **Run / handoff** | Run the selected local Apple gates or transfer a CUDA bundle to its intended host. |

### A concrete example

This repository includes a four-row synthetic support dataset. We profiled it
locally, paired it with declared 7B model facts and one 24 GiB CUDA device, then
ran the real planner. Under its quality policy, Aptus reports:

- **8-bit LoRA:** recommended, with a 13.9 GiB heuristic upper estimate against
  22 GiB usable device memory;
- **Full fine-tuning:** infeasible, with a 101.8 GiB upper estimate;
- **LoRA:** conditional because its 23.3 GiB upper estimate exceeds usable
  device memory; and
- **QLoRA:** feasible as the lower-memory alternative at 8.57 GiB.

The dataset profile and planning decision are real. The model and hardware facts
are declared examples. Target-host model loading, measurement, and pilot gates
can still reject the plan.

## Start on macOS

The current distribution supports local source builds and downloadable CI
artifacts for Apple Silicon. macOS 26 is the primary development and release
environment. macOS 15 is the fallback floor. A local build also needs Xcode 26,
XcodeGen, Node.js, `uv`, and Python 3.12 available to `uv`.

```bash
git clone https://github.com/ogprotege/Aptus.git
cd Aptus
desktop/macos/build.sh
open desktop/macos/dist/Aptus.app
```

The build runs the Python, React, and native test gates before creating:

```text
desktop/macos/dist/Aptus.app
desktop/macos/dist/Aptus-macOS-arm64.dmg
```

Every pull request and push to `main` also runs the native build on GitHub's
arm64 macOS 26 runner. The workflow uploads a permissions-preserving application
ZIP, the DMG, their `SHA256SUMS`, and a `COMMIT` source marker under an
`aptus-macos-arm64-<commit>` artifact. These CI artifacts use an ad-hoc
signature for review and testing. A public distribution still requires a
Developer ID signature and notarization.

See [installation details](docs/getting-started/install.md#build-aptus-for-mac)
for prerequisites, signing options, persistent paths, and the browser-based
development path.

## What runs where

| Native Mac product | MLX-LM runtime on Apple Silicon | CUDA runtime |
| --- | --- | --- |
| Inspect the Apple Silicon machine, models, data, plans, and runs | Verify exact MLX and MLX-LM versions | Verify Torch, Transformers, PEFT, and CUDA |
| Profile data and inspect pinned model facts | Load the pinned revision and tokenize all bound rows | Load the pinned revision and tokenizer |
| Compare runtime-specific estimates | Run a bounded adapter smoke with MLX memory telemetry | Run synthetic preflight, two-phase pilot, and admitted training |
| Compile, validate, and reveal artifacts | Run an uninterrupted pilot, reload its adapter in a fresh process, then admit confirmed full-duration adapter training | Produce and verify the selected export |

Choose the exact Python interpreter in the Mac Models screen. Aptus probes it,
persists the canonical path privately, and never treats the bundled backend or
an inference server as a training environment. CUDA profiles still describe a
CUDA host. They do not enable CUDA work on the Mac.

## Supported now

- CUDA supervised fine-tuning with Full, LoRA, int8-LoRA, and QLoRA.
- Conditional Apple Silicon MLX-LM LoRA and QLoRA planning and compilation.
- Uninterrupted MLX-LM pilot and full-duration adapter training. Pilot requires
  at least two optimizer updates, finite train and validation losses, an exact
  target census, positive memory and adapter-delta evidence, immutable
  artifacts, and fresh-process adapter reload with one to four generated tokens.
- Runtime-bound plans with separate compute, compiler, estimator, evidence, and
  export identities.
- Apple platform discovery for macOS/build, chip, CPU count, unified memory,
  current headroom, pressure, swap, Metal working-set guidance, optional Metal
  GPU core count, and ML runtime facts when the host can measure them.
- Apple planning keeps `free_vram_bytes` unknown and uses current free host RAM
  as the live unified-memory headroom cap when that measurement exists.
- Local LM Studio and oMLX model listing and generation adapters for inference
  and evaluation. Neither service is a training engine.
- JSON, JSONL, CSV, and text datasets using common SFT row shapes.
- Single-device and DDP plans where capability and memory rules pass.
- Conditional LoRA FSDP plans that still require a real multi-rank pilot.
- Explicit memory components, decision traces, evidence records, and artifact
  manifests.

Not yet supported: crash resume for MLX-LM or CUDA full runs, full-parameter or
DoRA training through MLX-LM, PyTorch MPS compilation, ROCm or CPU training,
CUDA execution on macOS, sequence packing, tasks other than SFT, and a
notarized public download. Read the
[complete capability matrix](docs/reference/capability-matrix.md) before
committing compute time.

## Why Aptus fails closed

- Unsupported combinations remain visible with their rejection reasons.
- Estimates never become measured facts merely because they rank first.
- Every compiled artifact binds back to its plan, candidate, data, and evidence.
- Pilot and train admission repeat checks against the current environment and
  available capacity.

## Data safety

Compiled bundles and ZIPs contain cleartext copies of training data. Runtime
artifacts can add model caches, logs, CUDA checkpoints, MLX weight snapshots,
metrics, adapters, and final weights.
Treat the entire bundle as sensitive. Read the [security policy](SECURITY.md)
before using private or governed data.

## Go deeper

| Goal | Documentation |
| --- | --- |
| Create a first plan without a GPU | [First-plan tutorial](docs/getting-started/first-plan.md) |
| Prepare real training data | [Dataset guide](docs/guides/prepare-a-dataset.md) |
| Operate an Apple or CUDA bundle | [Operator checklist](docs/operations/operator-checklist.md) |
| Understand the system | [Architecture](docs/architecture/system.md) |
| Integrate with Aptus | [CLI](docs/reference/cli.md) and [API](docs/reference/api.md) |
| Change the project | [Contributing](CONTRIBUTING.md) |

The complete documentation hub is [docs/index.md](docs/index.md).

## License

MIT. See [LICENSE](LICENSE).
