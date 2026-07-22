# Aptus

> **Status:** Active | **Authority:** Product entry point | **Applies to:** Aptus 0.2 | **Audience:** Users, operators, and contributors | **Last reviewed:** 2026-07-22 | **Review by:** 2026-10-22 or on a support-contract change

Aptus is an evidence-backed fine-tuning planner and artifact compiler. Give it
explicit model, dataset, hardware, and training-target facts. It compares the
strategies its current catalog can represent, explains assumptions and
tradeoffs, and emits a validated, ready-to-execute training bundle.

The product goal is simple: remove the repeated compatibility, memory,
configuration, and orchestration work that makes fine-tuning slow to start and
hard to reproduce. Aptus does not hide uncertainty. It turns assumptions into
named evidence gates and refuses combinations it cannot support.

Aptus 0.2 is an engineering preview. Its planner and compiler are usable now.
Its CUDA execution paths remain unreleased until the
[release gates](docs/operations/release-gates.md) have real target-host
evidence.

## What Aptus produces

One planning request produces three connected outputs:

1. **A decision record.** Every candidate retains its status, rejection or
   conditional reason, memory ledger, ranking basis, and evidence references.
2. **An identity-bound plan.** Model, data, hardware, target, strategy, and
   formula versions determine the plan and candidate identities.
3. **A portable training bundle.** The compiler writes validated data, direct
   package pins, configuration, generated Python entry points, a runbook, and a
   hash manifest to a new no-clobber directory.

Runtime checks then strengthen the evidence in order:

```text
plan -> compile -> static -> dependency -> model-data -> preflight -> pilot
     -> train admission -> unique full run -> parent verification
```

No planning estimate becomes a measured fact merely because a candidate ranks
first. No child process can mark its own artifacts complete.

## Current support snapshot

| Area | Current Aptus 0.2 contract |
| --- | --- |
| macOS application | Native AppKit host with authenticated bundled service and React workbench |
| Objective | Supervised fine-tuning only |
| Selectable methods | Full, LoRA, int8-LoRA, and QLoRA |
| Visible nonselectable methods | DoRA, BitFit, AdaLoRA, ShareLoRA, LoReFT, AFLoRA, and BiLoRA |
| Placement | Single CUDA device and DDP where feasible; LoRA FSDP is conditional |
| Execution backend | CUDA only |
| Apple Silicon | Hardware inventory only; MPS and MLX execution are not implemented |
| Input files | JSON, JSONL, CSV, and text |
| Task data | Text, prompt-completion, instruction-output, and chat-message SFT rows |
| Validation | Contract, static, dependency, model-data, measured preflight, and pilot |
| Full-run resume | Unsupported and fail-closed |
| Export proof | Structural safetensors and provenance checks, not quality evaluation |

See the normative [capability matrix](docs/reference/capability-matrix.md) for
the detailed method, placement, precision, and backend rules.

## Choose your starting path

### I want to use Aptus on this Mac

Build the native application, then open it from Finder:

```bash
desktop/macos/build.sh
open desktop/macos/dist/Aptus.app
```

The app starts its private planning service, opens the five-stage workbench,
provides native file and folder pickers, and stores state under
`~/Library/Application Support/Aptus`. It can profile data, compare methods,
compile a bundle, and run static validation on this Mac. It hands CUDA runtime
actions to the target host because macOS CUDA execution is not supported.

[Install Aptus for Mac](docs/getting-started/install.md#build-aptus-for-mac)

### I want a planning-only tutorial

Run the planning-only tutorial. It profiles bundled synthetic data, creates a
real plan, compiles a bundle, and validates it statically. It downloads no
model, allocates no accelerator memory, and starts no training process.

[Run the first-plan tutorial](docs/getting-started/first-plan.md)

### I want to use the browser workbench

Install the server extra and start the same-origin API and React application:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[server,test]'
aptus serve --host 127.0.0.1 --port 8787
```

Open `http://127.0.0.1:8787`. Keep the service on loopback. The jobs API is a
trusted-user local interface that can read files and launch processes. It has
no authentication boundary.

The workbench uses five stages:

1. Enter and inspect facts.
2. Compare viable, conditional, infeasible, and unsupported candidates.
3. Compile the recommended candidate.
4. Validate the exact bundle.
5. Run the ordered target-host actions.

### I have a CUDA host and want to train

Start with the [quickstart](docs/getting-started/quickstart.md), then use the
[operator checklist](docs/operations/operator-checklist.md). Replace every
example model and hardware value with measured facts for the intended host.

The managed action sequence is:

```bash
aptus run ./work/bundle --action dependency
aptus run ./work/bundle --action model-data
aptus run ./work/bundle --action preflight
aptus run ./work/bundle --action pilot
aptus run ./work/bundle --action train --confirm-full-train
```

Launch the sequence from an external environment that contains both Aptus and
the generated bundle requirements. The quickstart creates that environment.
Wait for each action to complete. Inspect state with `aptus jobs` and
`aptus jobs --id JOB_ID`. Train submission repeats deep authorization against
current artifacts, environment, CUDA identity, free VRAM, host RAM, and disk.

## The decisions Aptus makes

Aptus resolves a bounded planning problem. It does not perform unconstrained
hyperparameter search.

- It reads only the four selectable descriptors from the runtime method
  registry.
- It forms twelve visible rows from four methods and three placements.
- It applies model, task, backend, precision, quantization, batch, memory, host
  RAM, disk, and distribution rules.
- It calculates named point and upper memory components.
- It marks viable Pareto-frontier rows.
- It ranks feasible rows before conditional rows under the requested memory,
  speed, or quality policy.
- It preserves every unsupported row and reason in the plan.

The recommendation means “highest-ranked within the enumerated Aptus 0.2
candidate set.” It never means universally optimal, guaranteed to fit, or
guaranteed to improve the model.

## The evidence Aptus requires

Planning uses explicit and inferred facts. Runtime validation adds stronger
evidence without rewriting planning history.

| Stage | What Aptus checks | What it still does not prove |
| --- | --- | --- |
| Static | Contracts, identities, generated source, paths, hashes, direct pins | Imports, model load, CUDA fit |
| Dependency | Exact direct pins and resolved environment binding | Model or data compatibility |
| Model-data | Pinned model facts, target modules, method scope, trainable census, every canonical row | Optimizer behavior or planned-model fit |
| Measured preflight | Synthetic selected-method forward, backward, optimizer step, CUDA peak | Exact model and real-data behavior |
| Pilot | Two fresh real-model phases, checkpoint continuation, artifacts, measured peaks | Full-run completion or task quality |
| Measured run | Parent-verified metrics, split evidence, census, ranks, and export tree | Benchmark quality, safety, or deployment fitness |

Read [validation states](docs/reference/validation-states.md) for the complete
state and binding contract.

## Dataset and artifact safety

Compilation creates cleartext copies of the dataset in the bundle and its ZIP.
Runtime can add model caches, logs, checkpoints, metrics, tokenizer files, and
final model or adapter artifacts. Treat all of them as sensitive.

Related rows can declare `split_group`. The generated full trainer keeps each
declared group entirely in train or evaluation, records the requested and
realized evaluation sizes, and binds canonical and assignment digests. A large
indivisible group can prevent an exact requested fraction.

Review [dataset schemas](docs/reference/dataset-schemas.md), the
[reviewed-corpus contract](docs/reference/reviewed-corpus-contract.md), and the
[security policy](SECURITY.md) before using private or governed data.

## Repository map

```text
src/aptus/       planner, compiler, validators, API, CLI, and execution service
web/             React workbench source
desktop/macos/   native AppKit host, packaging, and macOS tests
docs/            current product, methodology, architecture, reference, and operations docs
Reference/       retained research inputs with explicit non-normative status
examples/        synthetic datasets and examples
tests/           Python and workbench contract tests
tools/           legacy recovery-audit tooling
```

The [code map](docs/architecture/code-map.md) connects each source module to its
contract, tests, and documentation owner.

## Documentation by reader

| If you are... | Start here |
| --- | --- |
| New to Aptus | [Choose your path](docs/getting-started/choose-your-path.md) |
| Building the Mac app | [Install Aptus](docs/getting-started/install.md#build-aptus-for-mac) |
| Running a first local check | [First plan](docs/getting-started/first-plan.md) |
| Preparing real training data | [Prepare a dataset](docs/guides/prepare-a-dataset.md) |
| Choosing a method | [Method selection guide](docs/guides/choose-a-method.md) |
| Operating a CUDA run | [Operator checklist](docs/operations/operator-checklist.md) |
| Inspecting results | [Inspect results](docs/guides/inspect-results.md) |
| Integrating the API | [API reference](docs/reference/api.md) |
| Extending Aptus | [Contributor documentation](docs/contributing/index.md) |
| Comparing research methods | [Research index](docs/research/index.md) |
| Maintaining the docs | [Documentation policy](docs/maintenance/documentation-policy.md) |

The complete navigation hub is [docs/index.md](docs/index.md).

## Development and verification

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and the full local quality
gate. Documentation changes must update the canonical contract page, related
examples, user-facing help, and drift tests together.

Current release evidence is intentionally incomplete. No real CUDA pilot has
been completed on this development Mac. The exact remaining proof is listed in
[release gates](docs/operations/release-gates.md).

## Project records

- [Changelog](CHANGELOG.md)
- [Roadmap](ROADMAP.md)
- [Security policy](SECURITY.md)
- [Support](SUPPORT.md)
- [Contributing](CONTRIBUTING.md)
- [Documentation health](docs/maintenance/documentation-health.md)
- [Documentation debt](docs/maintenance/documentation-debt.md)

## License

MIT. See [LICENSE](LICENSE).
