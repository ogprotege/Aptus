# Compile, Validate, and Run

> **Status:** Active | **Authority:** Operational execution guide | **Applies to:** Aptus 0.2 | **Audience:** CUDA operators | **Last reviewed:** 2026-07-22 | **Review by:** 2026-10-22 or when runtime actions change

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
so an earlier pass is both an admission prerequisite and a recorded checkpoint
in the operator workflow.

Model-data validation prepares the selected method before any optimizer exists.
It rejects zero or non-finite trainable parameters. The `full` path requires
every model tensor to remain trainable. LoRA, int8-LoRA, and QLoRA permit only
the compiled LoRA tensors to be trainable and require exactly one A/B pair for
every inspected target instance. Optimizer construction then proves its parameter
identities equal the validated trainable set. Measured preflight records the same
census contract for the synthetic method path. Both real-model pilot phases must
record identical census objects.

## Authorize full training

A pilot pass is necessary but not sufficient. At train submission, Aptus deeply
rechecks:

- compiler manifest and plan identity;
- source dataset and pinned model revision;
- installed environment binding;
- preflight and pilot metrics, including the selected method's trainable census;
- pilot checkpoint trees and pilot export trees;
- current CUDA identities and free VRAM;
- current free host RAM;
- current free disk using measured checkpoint and export sizes.

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

Direct portable child execution is supported on POSIX. On Windows, use the
managed `aptus run` command because the portable process-group contract is
fail-closed in v0.2.

Do not launch `train.py` directly. It requires the shared lease token. `run.py`
is the parent that chooses the
interpreter-bound single or Accelerate command, waits for aggregate completion,
verifies pending artifacts, and promotes the report.

The full trainer computes one deterministic split over the complete canonical
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

## Completion contract

Each full run receives a unique `run_*` directory. Aptus does not overwrite it.
The child writes pending metrics and export evidence while the validation report
stays `execution-approved`. The parent then verifies:

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

Only a successful parent verification promotes the report to
`measured-run-pass`. This proves the stated run and file-tree contract. It does
not prove task quality.

## Related documentation

- [Operator checklist](../operations/operator-checklist.md)
- [Validation states](../reference/validation-states.md)
- [Bundle manifest](../reference/bundle-manifest.md)
- [Recovery and resume boundary](resume-recover.md)
- [Inspect results](inspect-results.md)
