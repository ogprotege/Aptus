<p align="center">
  <img src="desktop/macos/Resources/AptusMark.svg" width="96" alt="Aptus mark">
</p>

<h1 align="center">Aptus</h1>

<p align="center"><strong>Plan a fine-tuning run on your Mac before you spend GPU time.</strong></p>

<p align="center">
  Aptus turns explicit model, dataset, hardware, and training facts into a ranked plan<br>
  and a statically validated bundle for measured checks on a CUDA host.
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

Aptus plans, compiles, and statically validates on macOS. Measured validation
and training run on the intended CUDA host. The Mac build is locally signed,
not notarized for public distribution. No real CUDA pilot has completed the
release gates yet.

</details>

<p align="center"><a href="docs/assets/aptus-macos-compare.jpeg"><img src="docs/assets/aptus-macos-compare.jpeg" width="1200" alt="Aptus for Mac comparing fine-tuning candidates and showing an 8-bit LoRA recommendation"></a></p>

<p align="center"><sub>Actual local planner output using the public synthetic dataset and declared example model and hardware facts. No model load or CUDA validation ran.</sub></p>

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
| **Run / handoff** | Hand the ordered dependency, model-data, preflight, pilot, and training commands to the CUDA host. |

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

The current distribution is a source build for Apple Silicon. You need macOS
13 or newer, Xcode, XcodeGen, Node.js, `uv`, and Python 3.12 available to `uv`.

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

See [installation details](docs/getting-started/install.md#build-aptus-for-mac)
for prerequisites, signing options, persistent paths, and the browser-based
development path.

## What runs where

| On your Mac | On the CUDA host |
| --- | --- |
| Choose and profile data | Bind the exact dependency environment |
| Enter or inspect pinned model facts | Load and verify the pinned model and tokenizer |
| Compare supported strategies | Measure a synthetic optimizer step |
| Compile and statically validate the bundle | Run the two-phase pilot |
| Reveal the directory and ZIP in Finder | Admit and launch the full training run |

The desktop service enforces this boundary server-side. Entering a remote CUDA
profile never enables local training controls on macOS.

## Supported now

- Supervised fine-tuning with Full, LoRA, int8-LoRA, and QLoRA.
- JSON, JSONL, CSV, and text datasets using common SFT row shapes.
- Single-device and DDP plans where capability and memory rules pass.
- Conditional LoRA FSDP plans that still require a real multi-rank pilot.
- Explicit memory components, decision traces, evidence records, and artifact
  manifests.

Not yet supported: MPS, MLX, ROCm, or CPU training; CUDA execution inside the
Mac app; sequence packing; full-training resume; tasks other than SFT; and a
notarized public download. Read the [complete capability matrix](docs/reference/capability-matrix.md)
before committing GPU time.

## Why Aptus fails closed

- Unsupported combinations remain visible with their rejection reasons.
- Estimates never become measured facts merely because they rank first.
- Every compiled artifact binds back to its plan, candidate, data, and evidence.
- Pilot and train admission repeat checks against the current environment and
  available capacity.

## Data safety

Compiled bundles and ZIPs contain cleartext copies of training data. Runtime
artifacts can add model caches, logs, checkpoints, metrics, and final weights.
Treat the entire bundle as sensitive. Read the [security policy](SECURITY.md)
before using private or governed data.

## Go deeper

| Goal | Documentation |
| --- | --- |
| Create a first plan without a GPU | [First-plan tutorial](docs/getting-started/first-plan.md) |
| Prepare real training data | [Dataset guide](docs/guides/prepare-a-dataset.md) |
| Operate a CUDA run | [Operator checklist](docs/operations/operator-checklist.md) |
| Understand the system | [Architecture](docs/architecture/system.md) |
| Integrate with Aptus | [CLI](docs/reference/cli.md) and [API](docs/reference/api.md) |
| Change the project | [Contributing](CONTRIBUTING.md) |

The complete documentation hub is [docs/index.md](docs/index.md).

## License

MIT. See [LICENSE](LICENSE).
