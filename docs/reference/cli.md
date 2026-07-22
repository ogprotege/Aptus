# CLI Reference

| Metadata | Value |
| --- | --- |
| Status | Active |
| Audience | Local operators, developers, and automation authors |
| Authority | Normative reference for the Aptus v0.2 command-line contract |
| Last reviewed | 2026-07-22 |
| Next review | 2026-10-22, or sooner when `src/aptus/cli.py` changes |

The `aptus` executable is installed from `aptus.cli:main`. Commands write JSON
to standard output unless an explicit output file is described below. JSON is
formatted with sorted keys and rejects non-finite numbers. Human-readable
errors use the `Aptus error:` prefix on standard error.

Run `aptus COMMAND --help` for the exact options in the installed build.

## Command summary

| Command | Purpose | Default action | Persistent side effects |
| --- | --- | --- | --- |
| `profile` | Profile a local dataset | Sample up to 512 valid rows for length statistics | Optional JSON output file |
| `spec-plan` | Write a standalone v2 plan | Objective `memory` | Plan JSON output file |
| `plan` | Compatibility flow for plan, compile, static validation, and archive | Same as `build` | Bundle, ZIP, optional plan file |
| `build` | Plan, compile, static validation, and archive | Objective `memory` | Bundle, ZIP, optional plan file |
| `compile` | Compile a persisted plan | Archive beside bundle | Bundle, ZIP, validation report |
| `validate` | Validate one evidence level | `static`, direct | Validation report and lock; runtime artifacts when executed |
| `run` | Submit a managed runtime action and wait | `preflight` | State records, log, report, and runtime artifacts |
| `jobs` | List or inspect managed jobs | List all | May reconcile stale persisted state |
| `hardware` | Inspect local hardware | Local probe | None |
| `inspect hardware` | Alias for `hardware` | Local probe | None |
| `inspect model` | Inspect bounded provider metadata | 10-second timeout | Provider network requests only |
| `serve` | Serve the API and packaged workbench | `127.0.0.1:8787` | State directories and job records as requests arrive |

## Planning fact options

`spec-plan`, `plan`, and `build` share the following facts.

### Model and dataset facts

| Option | Required | Default | Contract |
| --- | ---: | --- | --- |
| `--model-id ID` | Yes | None | Provider repository identifier, not a local path |
| `--revision HEX` | Yes | None | Immutable 40 to 64 character hexadecimal commit |
| `--family FAMILY` | Yes | None | Current adapter catalog: `llama`, `mistral`, `gemma`, or `qwen` |
| `--parameters-b NUMBER` | Yes | None | Positive parameter count in billions |
| `--hidden-size INTEGER` | Yes | None | Positive hidden width |
| `--intermediate-size INTEGER` | No | `null` | Positive MLP width; adapter estimates otherwise use `4 * hidden_size` |
| `--layers INTEGER` | Yes | None | Positive layer count |
| `--context-length INTEGER` | Yes | None | Positive model context limit |
| `--license LABEL` | Yes | None | User-supplied license label |
| `--confirm-training-allowed` | Yes in practice | False | Planning fails unless explicitly supplied |
| `--dataset PATH` | Yes | None | Existing `.jsonl`, `.json`, `.csv`, or `.txt` file |
| `--sample-limit INTEGER` | No | `512` | Positive bound for sampled length statistics, not canonical compilation |

### Hardware facts

| Option | Required | Default | Contract |
| --- | ---: | --- | --- |
| `--backend {cuda,rocm,mps,cpu}` | No | `cuda` | Known value only; v0.2 execution remains CUDA-only |
| `--gpu-count INTEGER` | Yes | None | Positive homogeneous manual device count |
| `--vram-gib NUMBER` | Yes | None | Positive total memory for every declared device |
| `--free-vram-gib NUMBER` | No | `null` | Current free memory; planner uses total when omitted |
| `--bf16` | No | False | Declares BF16 capability |
| `--four-bit` | No | False | Declares four-bit capability |
| `--eight-bit` | No | False | Declares eight-bit capability |
| `--host-ram-gib NUMBER` | Yes | None | Positive total host RAM |
| `--host-ram-free-gib NUMBER` | No | `null` | Current free host RAM; planner uses total when omitted |
| `--reserve-gib NUMBER` | No | `2.0` | Non-negative reserve subtracted from each device |
| `--disk-free-gib NUMBER` | No | `null` | Current free disk; analytic disk rejection is skipped when omitted |

Manual facts are recorded as `user-attested`. Repeating one set of values for
`--gpu-count N` creates `N` homogeneous device records. It does not scan the
host.

### Training target facts

| Option | Required | Default | Contract |
| --- | ---: | --- | --- |
| `--objective {quality,memory,speed}` | No | `memory` | Lexicographic ranking policy |
| `--sequence-length INTEGER` | Yes | None | Positive training sequence limit |
| `--effective-batch-size INTEGER` | No | `16` | Requested exact global batch |
| `--epochs INTEGER` | No | `3` | Positive full-run epoch count |
| `--prefer-method {full,lora,int8-lora,qlora}` | No | `null` | Tie-breaking preference, not a hard method constraint |
| `--evaluation-fraction NUMBER` | No | `0.1` | Value in `[0, 1)` |
| `--checkpoint-steps INTEGER` | No | `100` | Positive save and evaluation interval |
| `--packing` | No | False | Accepted as a fact, then rejected by the v0.2 planner |

The CLI fixes `task` to `sft`. It exposes no maximum wall-time field and no
full-training resume field.

## `aptus profile`

```bash
aptus profile --dataset DATASET [--sample-limit 512] \
  [--sequence-length TOKENS] [--output PROFILE.json]
```

Without `--output`, the profile is printed. With `--output`, parent directories
are created and an existing file is replaced. The command reads every row for
schema validation, counts, digesting, duplicate detection, and totals. The
sample limit bounds only percentile statistics and sample indices.

## `aptus spec-plan`

```bash
aptus spec-plan FACT_OPTIONS --output PLAN.json
```

The command profiles the source, constructs user-attested model and hardware
facts, enumerates the 12 candidates, and writes one `aptus.training-plan.v2`
document. Parent directories are created. An existing plan output file is
replaced. No bundle or archive is created.

## `aptus plan` and `aptus build`

```bash
aptus build FACT_OPTIONS --output BUNDLE [--plan-output PLAN.json]
```

Both names execute the same compatibility flow:

1. build the plan;
2. optionally write `--plan-output`;
3. compile into `--output`;
4. run static validation in the temporary bundle;
5. atomically publish the bundle;
6. create a deterministic ZIP beside it; and
7. print `bundle_dir`, `archive_path`, and the static `report`.

The bundle destination must be absent or empty. The ZIP must not exist. The
optional standalone plan is written before compilation and can therefore remain
when a later compilation step fails.

## `aptus compile`

```bash
aptus compile --plan PLAN.json --output BUNDLE [--archive BUNDLE.zip]
```

The plan is rehydrated through the v2 domain contract. The default archive is
the bundle path with its suffix replaced by `.zip`. The archive must be outside
the bundle. Bundle and archive publication are no-clobber. On success the
command prints the bundle path, archive path, and static validation report.

Compilation copies cleartext data and writes a mutable
`validation-report.json` to the directory. The deterministic ZIP excludes that
report and all runtime-only outputs.

## `aptus validate`

```bash
aptus validate BUNDLE \
  [--level {contract,static,dependency,model-data,measured-preflight,pilot}] \
  [--run] [--state-dir .aptus-state]
```

Default level is `static`. `contract` and `static` run directly, print the
report, and update `validation-report.json`. For a runtime level:

- without `--run`, direct validation records warning `RUNTIME_NOT_EXECUTED` and
  remains at `static-pass`;
- with `--run`, the CLI maps the level to a managed action, submits it, polls
  every 250 milliseconds, and prints the terminal job record.

The runtime mappings are `dependency` to `dependency`, `model-data` to
`model-data`, `measured-preflight` to `preflight`, and `pilot` to `pilot`.
Managed state is stored under `STATE_DIR/jobs`.

## `aptus run`

```bash
aptus run BUNDLE \
  [--action {dependency,model-data,preflight,pilot,train}] \
  [--confirm-full-train] [--state-dir .aptus-state]
```

Default action is `preflight`, but the prerequisite ladder still applies. The
command submits one managed job, waits, then prints the reconciled terminal
record. `--confirm-full-train` is valid only for `train` and is mandatory for
that action. There is no full-run resume option.

Pressing Control-C requests cancellation through the owning job service. The
CLI prints the resulting record when available and exits `130`.

## `aptus jobs`

```bash
aptus jobs [--state-dir .aptus-state] [--id JOB_ID]
```

Without an ID, the command prints persisted records in reverse creation order.
With an ID, it prints one reconciled record and its current validation report.
Reading a job is not a pure file read. Reconciliation can mark an unattached or
stale active record failed when its recorded process is no longer valid.

## `aptus hardware` and `aptus inspect hardware`

Both commands probe the local host and print this envelope:

```json
{
  "status": "ok",
  "scope": "server-local",
  "hardware": {}
}
```

CUDA hosts report visible devices and current capacity. Darwin arm64 without
CUDA reports one `mps` discovery record for shared unified memory. That record
does not enable MPS or MLX execution. Other unmeasurable hosts return
`status: unavailable`, `manual_facts_supported: true`, and exit `2`.

## `aptus inspect model`

```bash
aptus inspect model --model-id REPOSITORY \
  --revision REVISION [--timeout 10]
```

The timeout must be greater than zero and no more than 30 seconds. Inspection
makes bounded Hugging Face config and metadata requests. It returns status
`ok`, `unavailable`, or `unsupported`. Only `ok` exits successfully. Provider
metadata never decides parameter count or training permission.

## `aptus serve`

```bash
aptus serve [--host 127.0.0.1] [--port 8787] \
  [--state-dir .aptus-state] [--web-dist DIST] \
  [--allow-non-loopback]
```

The command requires the `server` optional dependency group. The packaged
workbench is used unless `--web-dist` names an explicit valid build. Binding to
anything other than `127.0.0.1`, `localhost`, or `::1` is rejected unless
`--allow-non-loopback` is present. That flag prints a warning and allows all
Host headers. It adds no authentication, bundle-root policy, or worker
isolation.

## Exit status

| Status | Meaning |
| ---: | --- |
| `0` | Command succeeded, report was not invalid, or managed job completed |
| `1` | Validation report is `invalid`, or managed job ended outside `completed` |
| `2` | Argument, input, inspection, JSON, filesystem, or command error |
| `130` | The CLI handled an interrupt by requesting job cancellation |

Argument-parser usage errors also exit `2`.

## Related documentation

- [Configuration defaults](configuration-defaults.md)
- [API reference](api.md)
- [Plan schema](plan-schema.md)
- [Bundle manifest](bundle-manifest.md)
- [Run states](run-states.md)
- [Validation states](validation-states.md)
