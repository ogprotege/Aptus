# Aptus

Aptus is an evidence-backed fine-tuning planner and artifact compiler. Give it
explicit model, dataset, hardware, and training-target facts. It compares the
strategies that its current catalog can represent, explains assumptions and
tradeoffs, and emits a portable training bundle.

Aptus v0.2 is an engineering preview. It does not promise a universally optimal
strategy, model quality, wall-clock time, cost, or VRAM fit. Analytic estimates
remain estimates until the selected bundle passes runtime checks on the target
host.

## What v0.2 does

- Validates every source training row from local JSON, JSONL, CSV, or text data
  and records a source digest.
- During full training, keeps related rows with the same explicit `split_group`
  on one side of the deterministic train and evaluation boundary. It records
  target and realized sizes because indivisible groups can prevent an exact
  requested fraction.
- Accepts an immutable model revision plus explicit architecture and permission
  facts.
- Enumerates full fine-tuning, LoRA, int8-LoRA, and QLoRA across supported
  single-device and distributed placements.
- Publishes a typed method-readiness catalog that separates those executable
  paths from experimental and research-only methods.
- Enforces a positive, finite, method-specific trainable-parameter census before
  optimizer construction. Adapter paths require one LoRA A/B pair for every
  inspected target instance. The optimizer parameter identities must then equal
  that validated set. Measured evidence records its name-shape-dtype digest.
- Produces point and upper memory estimates with cited assumptions.
- Compiles a no-clobber bundle containing data, plan, evidence, direct package
  pins, launch configuration, validators, and training code.
- Enforces five managed runtime actions in order: dependency, model-data,
  preflight, pilot, then train. Higher validation actions also recheck lower
  levels inside their job.
- Persists local managed jobs, streams logs, supports cancellation, and uses one
  host-global Aptus execution lease per user.
- Writes every full run under a unique run ID and promotes a completed run only
  after the parent process verifies metrics, bindings, and the structural export
  file tree.

## Current boundaries

- CUDA is the only execution backend in the v0.2 support contract. Other
  backends can appear as known values but are not execution-ready. Local Apple
  Silicon inspection records its shared unified-memory pool without treating
  it as dedicated VRAM or pretending the CUDA compiler can run there.
- Full fine-tuning requires BF16. Adapter methods can select FP16 when BF16 is
  not declared, subject to the exact pilot gate.
- Full-parameter FSDP is unsupported. LoRA FSDP is conditional. Quantized FSDP
  combinations are unsupported.
- Full-training resume is fail-closed. Pilot continuation is a bounded validation
  exercise, not a general resume feature.
- Model inspection reads bounded provider-declared metadata. The user must still
  supply and confirm license and training-permission facts.
- Evaluation targets, exporter plugins, cloud providers, and MCP adapters are
  future extension seams. They are not current execution features.

No real CUDA pilot has been completed on the current development Mac. The
repository is therefore not release-ready. See
[`docs/operations/release-gates.md`](docs/operations/release-gates.md).

## Install

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[server,test]'
```

The React source is in `web/`. A packaged build is already served from
`src/aptus/_web`.

## Start the workbench

```bash
aptus serve --host 127.0.0.1 --port 8787
```

Open `http://127.0.0.1:8787`. Keep the service on loopback. The jobs API is a
trusted-user local interface and has no authentication boundary.

The workbench follows five stages:

1. Enter and inspect facts.
2. Compare viable candidates, including explicitly conditional rows.
3. Compile a selected plan.
4. Validate the bundle.
5. Run the five ordered execution actions.

## CLI example

The repository includes `examples/support-sft.jsonl`. Replace the model facts and
hardware facts with measured values for the intended run.

```bash
aptus spec-plan \
  --model-id provider/model \
  --revision IMMUTABLE_COMMIT \
  --family llama \
  --parameters-b 7 \
  --hidden-size 4096 \
  --intermediate-size 11008 \
  --layers 32 \
  --context-length 4096 \
  --license LICENSE_NAME \
  --confirm-training-allowed \
  --dataset examples/support-sft.jsonl \
  --gpu-count 1 \
  --vram-gib 24 \
  --free-vram-gib 22 \
  --bf16 --four-bit --eight-bit \
  --host-ram-gib 64 \
  --host-ram-free-gib 48 \
  --disk-free-gib 200 \
  --objective memory \
  --sequence-length 1024 \
  --output ./work/plan.json

aptus compile --plan ./work/plan.json --output ./work/bundle
aptus validate ./work/bundle --level static
```

Use managed jobs for the runtime sequence:

```bash
aptus run ./work/bundle --action dependency
aptus run ./work/bundle --action model-data
aptus run ./work/bundle --action preflight
aptus run ./work/bundle --action pilot
aptus run ./work/bundle --action train --confirm-full-train
```

Wait for each action to finish successfully before starting the next. Inspect
state with `aptus jobs` or `aptus jobs --id JOB_ID`.

## Portable bundle execution

Inside a compiled bundle:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python validate.py --level dependency
python validate.py --level model-data
python validate.py --level measured-preflight
python validate.py --level pilot
python run.py --confirm-full-train
```

`requirements.txt` is the direct, method-specific pinned input set. It is not a
complete transitive lock file. Capture and retain the installed-environment
binding produced by validation.

`run.py` is the portable full-run parent. It launches the selected single or
distributed command, waits for aggregate completion, verifies the pending
artifacts, and promotes the validation report to `measured-run-pass` only when
those checks succeed. These direct portable commands are supported on POSIX.
On Windows, use the managed `aptus run` path because direct portable child
execution is fail-closed in v0.2.

## Documentation

- [Documentation index](docs/index.md)
- [Current capabilities](docs/product/current-capabilities.md)
- [Quickstart](docs/getting-started/quickstart.md)
- [System architecture](docs/architecture/system.md)
- [Validation states](docs/reference/validation-states.md)
- [Release gates](docs/operations/release-gates.md)
- [Apple Silicon pilot matrix](docs/operations/apple-silicon-pilot.md)
- [Reviewed corpus contract](docs/reference/reviewed-corpus-contract.md)
- [Security policy](SECURITY.md)

## License

MIT. See `LICENSE`.
