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
**Last reviewed:** 2026-07-27 | **Review by:** 2026-10-27 or when the support contract changes

Aptus has separate CUDA and MLX-LM compiler contracts. Apple Silicon LoRA and
QLoRA candidates remain conditional until their exact bundle passes measured
gates. Two clean, independent Apple Silicon workflows reached
`measured-run-pass` against a revision-pinned public model. Crash resume remains
unsupported. Ten consecutive clean local desktop engineering builds also passed
at implementation commit `1038ecdd13103418ef1135e1ced634c10370a961`. That
record is historical evidence for that exact commit. Pull-request CI rebuilds
and packages GitHub's exact tested merge commit. The artifact records that
commit in `COMMIT`. The default Mac build is ad-hoc signed.
Public distribution still requires a Developer ID identity and notarization.
No real CUDA target-host pilot has completed the release gates.
The first MoE compatibility slice is exact and fail-closed. It recognizes
`qwen3_moe` checkpoints with `Qwen3MoeForCausalLM` only when they use the
reviewed MLX layout: four-bit group-64 defaults plus one eight-bit group-64
`model.layers.N.mlp.gate` override per layer. It then permits only
single-device MLX-LM QLoRA with attention-only adapters. That MoE slice still
requires full real-model acceptance. Its first exact 30B attempt passed
dependency validation, then refused model loading with an 18.932 GiB live
unified-memory shortfall. See the
[Qwen3 MoE admission record](docs/operations/evidence/2026-07-28-qwen3-moe-admission/README.md).

</details>

## One decision, with the reasons attached

| You provide | Aptus evaluates | You receive |
| --- | --- | --- |
| A pinned model, supervised dataset, target hardware, and training goal | Which supported methods appear feasible, which fail, and why | A decision record and a no-clobber training bundle with code, configuration, evidence, and hashes |

Aptus compares Full, LoRA, int8-LoRA, and QLoRA across the placements it can
represent. It preserves infeasible and unsupported candidates instead of hiding
them. A recommendation means highest-ranked within this bounded Aptus catalog.
It is not a quality prediction or a guarantee that unmeasured hardware will fit.

---

## The five-stage workbench

| Stage | What happens |
| --- | --- |
| **Facts** | Profile local data and record the exact model, hardware, and training target. |
| **Compare** | Apply hard feasibility rules, inspect memory ledgers, and explain the ranking. |
| **Compile** | Write a new bundle directory and deterministic ZIP without overwriting prior work. |
| **Validate** | Check contracts, identities, generated source, paths, hashes, and direct dependency pins. |
| **Run / handoff** | Run the selected local Apple gates or transfer a CUDA bundle to its intended host. |

The native Mac window presents one product surface. Its sidebar owns Home,
Workbench, Machine, and Models. The React workbench stays inline and owns the
single Facts, Compare, Compile, Validate, and Run workflow. Named projects keep
immutable revisions for plans, bundles, validation, and jobs. Recovering an old
revision creates a new revision and never restores training authorization.

---

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

The same synthetic dataset later exercised the complete MLX-LM QLoRA runtime
against a revision-pinned public model. Two clean workflows reached
`measured-run-pass`. That result proves the recorded runtime and artifact
contracts. It is not a model-quality benchmark. See the
[MLX-LM acceptance record](docs/operations/evidence/2026-07-27-mlx-lm-acceptance/README.md).

### Recorded acceptance snapshot

| Exact recorded gate | Observed result |
| --- | ---: |
| MLX-LM five-action workflow | 18.65 s and 17.47 s, 18.06 s mean |
| Confirmed full train, export, and fresh reload | 4.73 s and 5.06 s |
| Highest full-run MLX peak | 555.1 MiB |
| Qwen3 30B MoE live admission | 47.759 GiB required, 28.827 GiB available, 18.932 GiB shortfall |
| Real MLX synthetic MoE forward | 0.877 ms median for a small unquantized two-layer probe |
| Ten clean desktop builds at `1038ecdd13103418ef1135e1ced634c10370a961` | 58.1 s mean, 55 to 63 s range |

The MLX figures are acceptance telemetry for the recorded M5 Pro host, 0.5B
four-bit model, and four-row synthetic dataset. They are not production
throughput, scalability, or model-quality measurements. The desktop timing is
historical evidence for its exact implementation commit.
The synthetic MoE forward is not autoregressive generation and does not project
30B speed. The 30B checkpoint never loaded, so no 30B throughput claim exists.

---

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
desktop/macos/dist/Aptus.app.zip
desktop/macos/dist/Aptus-macOS-arm64.dmg
desktop/macos/dist/SHA256SUMS
desktop/macos/dist/COMMIT
```

Every pull request and push to `main` also runs the native build on GitHub's
arm64 macOS 26 runner. The workflow uploads a permissions-preserving application
ZIP, the DMG, their `SHA256SUMS`, and a `COMMIT` source marker under an
`aptus-macos-arm64-<commit>` artifact. These CI artifacts use an ad-hoc
signature for review and testing. A public distribution still requires a
Developer ID signature and notarization.

The first local ten-build stability gate passed 10 of 10 clean builds at
implementation commit `1038ecdd13103418ef1135e1ced634c10370a961`. Its
[desktop engineering evidence](docs/operations/evidence/2026-07-27-desktop-release/README.md)
does not claim that later commits have the same binaries. Pull-request CI is the
merge-candidate package check. It records the workflow commit rather than
mislabeling it as the branch head.

See [installation details](docs/getting-started/install.md#build-aptus-for-mac)
for prerequisites, signing options, persistent paths, and the browser-based
development path.

---

## What runs where

| Native Mac product | MLX-LM runtime on Apple Silicon | CUDA runtime |
| --- | --- | --- |
| Inspect the Apple Silicon machine, models, data, plans, and runs | Verify exact MLX and MLX-LM versions | Verify Torch, Transformers, PEFT, and CUDA |
| Profile data and inspect pinned model facts | Load the pinned revision and tokenize all bound rows | Load the pinned revision and tokenizer |
| Compare runtime-specific estimates | Run a bounded adapter smoke with MLX memory telemetry | Run synthetic preflight, two-phase pilot, and admitted training |
| Compile, validate, and reveal artifacts | Run an uninterrupted pilot, reload its adapter in a fresh process, then admit confirmed full-duration adapter training | Produce and verify the selected export |

Choose the exact Python interpreter in the Mac Models screen. Its environment
doctor shows every likely interpreter, Python version, import result, and exact
pin-compatibility result. Aptus installs nothing. Only an interpreter matching
the reviewed MLX pins can be selected or reported ready. CUDA profiles describe
a CUDA host. They do not enable CUDA work on the Mac.

---

## Supported now

- CUDA supervised fine-tuning with Full, LoRA, int8-LoRA, and QLoRA.
- Conditional Apple Silicon MLX-LM LoRA and QLoRA planning and compilation.
- Exact mixed-precision Qwen3 MoE inspection and conditional single-device
  MLX-LM QLoRA planning. The reviewed checkpoint layout uses four-bit group-64
  defaults and one eight-bit group-64 router-gate override per layer. Aptus
  records routed-expert topology, derives active parameters and sparse-layer
  count, and keeps the user-attested total parameter count as the resident-weight
  budget. Other four-bit Qwen3 MoE layouts remain unsupported.
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
- Named local projects with immutable, content-hashed revisions and recovery
  that always requires fresh validation and training confirmation.
- Typed API responses under `aptus.api.v1`, plus a checked OpenAPI artifact at
  `docs/reference/openapi.v1.json`.
- Read-only runtime diagnosis through `aptus doctor` and privacy-bounded support
  archives through `aptus diagnostics`.

Not yet supported: crash resume for MLX-LM or CUDA full runs, full-parameter or
DoRA training through MLX-LM, PyTorch MPS compilation, ROCm or CPU training,
CUDA execution on macOS, general MoE families, MoE CUDA execution, shared-expert
Qwen3 MoE variants, MoE methods other than the exact MLX-LM QLoRA path,
sequence packing, tasks other than SFT, and a notarized public download. Read the
[complete capability matrix](docs/reference/capability-matrix.md) before
committing compute time.

---

## Why Aptus fails closed

- Unsupported combinations remain visible with their rejection reasons.
- Estimates never become measured facts merely because they rank first.
- Every compiled artifact binds back to its plan, candidate, data, and evidence.
- Pilot and train admission repeat checks against the current environment and
  available capacity.

---

## Data safety

Compiled bundles and ZIPs contain cleartext copies of training data. Runtime
artifacts can add model caches, logs, CUDA checkpoints, MLX weight snapshots,
metrics, adapters, and final weights.
Treat the entire bundle as sensitive. Read the [security policy](SECURITY.md)
before using private or governed data.

---

## Go deeper

| Goal | Documentation |
| --- | --- |
| Create a first plan without a GPU | [First-plan tutorial](docs/getting-started/first-plan.md) |
| Prepare real training data | [Dataset guide](docs/guides/prepare-a-dataset.md) |
| Operate an Apple or CUDA bundle | [Operator checklist](docs/operations/operator-checklist.md) |
| Review the real Apple acceptance | [MLX-LM acceptance evidence](docs/operations/evidence/2026-07-27-mlx-lm-acceptance/README.md) |
| Review desktop engineering stability | [Desktop release evidence](docs/operations/evidence/2026-07-27-desktop-release/README.md) |
| Understand the system | [Architecture](docs/architecture/system.md) |
| Integrate with Aptus | [CLI](docs/reference/cli.md) and [API](docs/reference/api.md) |
| Change the project | [Contributing](CONTRIBUTING.md) |

The complete documentation hub is [docs/index.md](docs/index.md).

---

## Funding

<a href="https://www.buymeacoffee.com/thebiscuit" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee" style="height: 60px !important;width: 217px !important;" ></a>

---

## License

MIT. See [LICENSE](LICENSE).
