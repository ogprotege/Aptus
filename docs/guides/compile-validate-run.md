# Compile, Validate, and Run

> **Status:** Active | **Authority:** Operational execution guide | **Applies to:** Aptus 0.2 | **Audience:** CUDA and Apple Silicon operators | **Last reviewed:** 2026-08-04 | **Review by:** 2026-10-22 or when runtime actions change

## Compile once

```bash
aptus compile --plan ./work/plan.json --output ./work/bundle
```

Compilation is atomic and no-clobber. The destination must be absent or empty.
The generated archive must not already exist. If facts or strategy change,
create a new plan and a new bundle path.

## Install the bundle stack

From the repository root, keep an absolute bundle path and create the runtime
environment beside the bundle:

```bash
BUNDLE_DIR="$(pwd)/work/bundle"
python -m venv ./work/bundle-env
source ./work/bundle-env/bin/activate
python -m pip install -e .
python -m pip install -r "$BUNDLE_DIR/requirements.txt"
```

`requirements.txt` contains exact direct pins for the selected method. It does
not enumerate every transitive package selected by the installer. Keep the
environment outside the bundle. An in-bundle `.venv` is an unexpected path and
invalidates the compiler manifest. Managed jobs use the interpreter that
launched `aptus`, so that environment must contain both Aptus and the bundle
stack. When operating from an installed wheel instead of a checkout, install
that Aptus wheel in place of the editable command above.

## Validate in order

Portable execution uses the same activated environment. A subshell keeps the
operator's working directory stable:

```bash
(
  cd "$BUNDLE_DIR"
  python validate.py --level static
  python validate.py --level dependency
  python validate.py --level model-data
  python validate.py --level measured-preflight
  python validate.py --level pilot
)
```

These package-free commands validate the canonical policy snapshot frozen into
the bundle. They prove the snapshot's schema, path, canonical bytes, digest
bindings, and decision parity with the saved plan. They cannot determine whether
an installed host's model-policy registry has advanced since compilation.

The repository CLI exposes the same runtime work as cancellable jobs:

```bash
aptus run "$BUNDLE_DIR" --action dependency
aptus run "$BUNDLE_DIR" --action model-data
aptus run "$BUNDLE_DIR" --action preflight
aptus run "$BUNDLE_DIR" --action pilot
```

Wait for each action. The local orchestrator allows one managed Aptus job at a
time across state roots for the same user. It rejects skipped actions. A higher
validation action also reruns the lower validation levels inside its own job,
so an earlier pass is both an admission prerequisite and a recorded state in the
operator workflow.

Managed execution adds the installed-host currency boundary. Aptus checks the
current registry at submission, pilot authorization, worker launch, and the
completion verification and promotion transaction. A standalone validation
pass does not waive those checks. When a coherent v5 plan's saved decision or
snapshot digest is no longer current, saved-plan load, compile, project
recovery, and managed job submission APIs return HTTP `409 replan_required`;
CLI and job surfaces include `replan_required`. Preserve the old artifact,
create a new plan from its source facts, and compile to a new bundle path.

CUDA model-data validation prepares the selected method before any optimizer
exists. It rejects zero or non-finite trainable parameters. The `full` path
requires every model tensor to remain trainable. LoRA, int8-LoRA, and QLoRA
permit only compiled LoRA tensors and require exactly one A/B pair per inspected
target instance. CUDA measured preflight records the same census for a synthetic
method path, and both real-model pilot phases must agree.

MLX model-data validation loads the pinned revision and tokenizes every bound
train and validation row. Measured preflight completes bounded real-input
optimizer work. Pilot runs the exact model and data from the pinned base without
interruption for at least two optimizer updates. It proves exact target binding,
finite losses, positive memory and adapter delta, live headroom, immutable
artifacts, and fresh-process adapter reload with one to four generated tokens.

## Authorize full training

A pilot pass is necessary but not sufficient. For CUDA, train submission deeply
rechecks:

- compiler manifest and plan identity;
- canonical policy-snapshot bytes and the snapshot, plan, manifest, and
  current-host digest bindings;
- the saved compatibility decision against the installed host's current
  model-policy registry;
- source dataset and pinned model revision;
- installed environment binding;
- preflight and pilot metrics, including the selected method's trainable census;
- pilot checkpoint trees and pilot export trees;
- current CUDA identities and free VRAM;
- current free host RAM;
- current free disk using measured checkpoint and export sizes.

For MLX, submission re-verifies the owned uninterrupted pilot, including its
adapter and reload manifests. It then requires current available unified memory
above the measured pilot peak plus reserve and enough free disk for the plan and
measured adapter artifacts.

This check runs inside the global lease and job-record transaction. Cached status
text is informational.

## Start full training

Managed execution:

```bash
aptus run "$BUNDLE_DIR" --action train --confirm-full-train
```

Portable execution with the same external environment:

```bash
(
  cd "$BUNDLE_DIR"
  python run.py --confirm-full-train
)
```

Direct portable execution remains bound to the bundle's frozen snapshot. Use
installed Aptus when current host policy must be established; do not interpret a
direct portable pass as current-registry authorization.

Direct portable child execution is supported on POSIX. On Windows, use the
managed `aptus run` command because the portable process-group contract is
fail-closed in v0.2.

Do not launch `train.py` directly. It requires the shared lease token. `run.py`
is the parent that chooses the
interpreter-bound single or Accelerate command, waits for aggregate completion,
verifies pending artifacts, and promotes the report.

The CUDA full trainer computes one deterministic split over the complete canonical
JSONL. Ungrouped data uses `deterministic-exact-row-count-sha256`. Data with at
least one declared `split_group` uses
`deterministic-size-aware-group-sha256`. Every declared group stays on one side.
The solver reaches the requested row count whenever a declared-group subset plus
the available ungrouped rows makes it attainable. Otherwise an indivisible group
can force a different realized size. The evidence records both sizes and the row
error.

The trainer hashes the canonical file during each split pass, binds every rank
to the same canonical and assignment digests, and rechecks the file before and
during lazy consumption. A change aborts the run.

The MLX compiler instead writes disjoint `data/mlx/train.jsonl` and
`data/mlx/valid.jsonl` files before runtime. It pads only within each split and
binds source and compiled counts in `aptus.mlx-split.v1`. The current MLX split
does not claim CUDA's group-aware assignment behavior.

## Completion contract

Each full run receives a unique `run_*` directory. Aptus does not overwrite it.
The child writes metrics and export evidence while the validation report remains
below final completion. For CUDA, the parent verifies:

- successful aggregate process exit;
- finite training and evaluation loss where required;
- positive completed step count;
- exact plan, candidate, run, distribution, world-size, and rank bindings;
- measured CUDA peaks;
- a positive finite trainable census with the selected method's exact scope and
  a valid descriptor digest;
- internally consistent dataset-split strategy, digests, counts, target size,
  realized fraction, and row error;
- expected safetensors files and nonempty tensor keys;
- adapter or full-model provenance;
- recursive path, size, and hash manifest coverage.

For MLX, the full run starts again from the pinned base and derives its
uninterrupted duration from the compiled train rows, batch, accumulation, and
epochs. The parent verifies at least one optimizer update, finite train and
validation losses, exact target binding, positive MLX peak and adapter delta,
live headroom evidence, immutable run artifacts, fresh-process one-to-four-token
generation, and `aptus.mlx-final-export.v1`. Weight snapshots cannot resume the
run.

Only a successful parent verification promotes the report to
`measured-run-pass`. This proves the stated run and file-tree contract. It does
not prove task quality. Managed promotion also rechecks current host model
policy, so evidence completed under a stale coherent v5 policy is not newly
promoted.

## Related documentation

- [Operator checklist](../operations/operator-checklist.md)
- [Validation states](../reference/validation-states.md)
- [Bundle manifest](../reference/bundle-manifest.md)
- [Recovery and resume boundary](resume-recover.md)
- [Inspect results](inspect-results.md)
