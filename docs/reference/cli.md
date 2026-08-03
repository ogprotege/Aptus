# CLI Reference

| Metadata | Value |
| --- | --- |
| Status | Active |
| Audience | Local operators, developers, and automation authors |
| Authority | Normative reference for the Aptus v0.2 command-line contract |
| Last reviewed | 2026-08-03 |
| Next review | 2026-11-01, or sooner when `src/aptus/cli.py` changes |

The `aptus` executable is installed from `aptus.cli:main`. Commands write JSON
to standard output unless an explicit output file is described below. JSON is
formatted with sorted keys and rejects non-finite numbers. Human-readable
errors use the `Aptus error:` prefix on standard error.

Run `aptus COMMAND --help` for the exact options in the installed build.

## Command summary

| Command | Purpose | Default action | Persistent side effects |
| --- | --- | --- | --- |
| `profile` | Profile a local dataset | Sample up to 512 valid rows for length statistics | Optional JSON output file |
| `spec-plan` | Write a standalone v5 plan | Objective `memory` | Plan JSON output file |
| `plan` | Compatibility flow for plan, compile, static validation, and archive | Same as `build` | Bundle, ZIP, optional plan file |
| `build` | Plan, compile, static validation, and archive | Objective `memory` | Bundle, ZIP, optional plan file |
| `compile` | Compile a persisted plan | Archive beside bundle | Bundle, ZIP, validation report |
| `validate` | Validate one evidence level | `static`, direct | Validation report and lock; runtime artifacts when executed |
| `run` | Submit a managed runtime action and wait | `preflight` | State records, log, report, and runtime artifacts |
| `jobs` | List or inspect managed jobs | List all | May reconcile stale persisted state |
| `doctor` | Inspect training-runtime readiness without changing it | Read `.aptus-state` and probe likely interpreters | Optional JSON output file; no installation |
| `diagnostics` | Create a privacy-bounded support archive | Summarize `.aptus-state` | New mode-0600 ZIP |
| `hardware` | Inspect local hardware | Local probe | None |
| `inspect hardware` | Alias for `hardware` | Local probe | None |
| `inspect model` | Inspect bounded provider metadata | 10-second timeout | Provider network requests only |
| `serve` | Serve an authenticated API and packaged workbench | `127.0.0.1:8787` | Per-launch workbench handoff, bearer token, state, and job records |

## Planning fact options

`spec-plan`, `plan`, and `build` share the following facts.

### Model and dataset facts

| Option | Required | Default | Contract |
| --- | ---: | --- | --- |
| `--model-id ID` | Yes | None | Provider repository identifier, not a local path |
| `--revision HEX` | Yes | None | Immutable 40 to 64 character hexadecimal commit |
| `--inspection-receipt PATH` | No | None | Successful `aptus inspect model` JSON or its nested receipt; every covered fact is revalidated |
| `--family FAMILY` | Yes | None | Dense adapter catalog or exact inspected `qwen3_moe` row |
| `--parameters-b NUMBER` | Yes | None | Positive total resident parameter count in billions; never substitute active MoE parameters |
| `--model-type TYPE` | No | `null` | Exact provider model type; required by allowlisted MoE contracts |
| `--architecture CLASS` | No | `null` | Exact provider architecture class; required by allowlisted MoE contracts |
| `--quantization-bits BITS` | No | `null` | Pinned checkpoint precision from 1 through 16 bits |
| `--quantization-layout-profile PROFILE` | No | `null` | Exact reviewed provider map; the first MoE row requires `qwen3-moe-4bit-group64-router-gates-8bit` |
| `--hidden-size INTEGER` | Yes | None | Positive hidden width |
| `--intermediate-size INTEGER` | No | `null` | Positive MLP width; adapter estimates otherwise use `4 * hidden_size` |
| `--moe-expert-count INTEGER` | For MoE | `null` | Positive total routed-expert count |
| `--moe-experts-per-token INTEGER` | For MoE | `null` | Positive experts selected per token |
| `--moe-expert-intermediate-size INTEGER` | For MoE | `null` | Positive routed-expert width |
| `--moe-decoder-sparse-step INTEGER` | For MoE | `null` | Positive sparse-layer cadence |
| `--moe-mlp-only-layer INTEGER` | No | Empty | Repeat for each zero-based dense-only layer; values must be sorted and unique |
| `--moe-shared-expert-intermediate-size INTEGER` | No | `null` | Optional positive shared-expert width; unsupported by the first executable row |
| `--layers INTEGER` | Yes | None | Positive layer count |
| `--context-length INTEGER` | Yes | None | Positive model context limit |
| `--license LABEL` | Yes | None | User-supplied license label |
| `--confirm-training-allowed` | Yes in practice | False | Planning fails unless explicitly supplied |
| `--dataset PATH` | Yes | None | Existing `.jsonl`, `.json`, `.csv`, or `.txt` file |
| `--sample-limit INTEGER` | No | `512` | Positive bound for sampled length statistics, not canonical compilation |

Without `--inspection-receipt`, model facts and the policy-decision source are
`user-attested`. A valid receipt changes only its covered provider-declared or
inferred model fields and sets the source to `provider-inspection`. Parameter
count and training permission remain user-attested and are excluded from the
receipt. A present but invalid receipt fails; it never silently falls back to
manual facts.

### Hardware facts

| Option | Required | Default | Contract |
| --- | ---: | --- | --- |
| `--backend {cuda,rocm,mps,cpu}` | No | `cuda` | Planned compute backend |
| `--gpu-count INTEGER` | Yes | None | Positive homogeneous manual device count |
| `--vram-gib NUMBER` | Yes | None | Positive total memory for every declared device |
| `--free-vram-gib NUMBER` | No | `null` | Current free memory; planner uses total when omitted |
| `--bf16` | No | False | Declares BF16 capability |
| `--four-bit` | No | False | Declares four-bit capability |
| `--eight-bit` | No | False | Declares eight-bit capability |
| `--host-ram-gib NUMBER` | Yes | None | Positive total host RAM |
| `--host-ram-free-gib NUMBER` | No | `null` | Current free host RAM; planner uses total when omitted |
| `--reserve-gib NUMBER` | No | `2.0` | Non-negative reserve subtracted from each device. Raised to at least `8.0` when `--backend mps` is selected |
| `--disk-free-gib NUMBER` | No | `null` | Current free disk; analytic disk rejection is skipped when omitted |

Manual hardware facts are recorded as `user-attested`. Repeating one set of
values for `--gpu-count N` creates `N` homogeneous device records. It does not
scan the host.

### Training target facts

| Option | Required | Default | Contract |
| --- | ---: | --- | --- |
| `--objective {quality,memory,speed}` | No | `memory` | Lexicographic ranking policy |
| `--training-runtime {transformers-peft-cuda,mlx-lm,pytorch-mps}` | No | `null` | Explicit runtime binding; must match `--backend` |
| `--sequence-length INTEGER` | Yes | None | Positive training sequence limit |
| `--effective-batch-size INTEGER` | No | `16` | Requested exact global batch |
| `--epochs INTEGER` | No | `3` | Positive full-run epoch count |
| `--prefer-method {full,lora,int8-lora,qlora}` | No | `null` | Tie-breaking preference, not a hard method constraint |
| `--evaluation-fraction NUMBER` | No | `0.1` | Value in `[0, 1)` |
| `--checkpoint-steps INTEGER` | No | `100` | Positive CUDA save and evaluation interval; MLX keeps it as a plan fact but uses non-resumable weight snapshots |
| `--packing` | No | False | Accepted as a fact, then rejected by the v0.2 planner |

When `--training-runtime` is omitted, the planner preserves legacy inference:
CUDA uses `transformers-peft-cuda`; MPS LoRA and QLoRA candidates use `mlx-lm`;
and other MPS methods use the known but unimplemented `pytorch-mps` binding.
An explicit `transformers-peft-cuda` selection requires `--backend cuda`.
Explicit `mlx-lm` and `pytorch-mps` selections require `--backend mps`.
`pytorch-mps` remains an implementation-required runtime, so it cannot produce
an executable candidate.

Supplying any MoE topology option requires all four required MoE options. The
first executable row also requires `--model-type qwen3_moe`,
`--architecture Qwen3MoeForCausalLM`, `--quantization-bits 4`, `--backend mps`,
`--quantization-layout-profile qwen3-moe-4bit-group64-router-gates-8bit`, an
MLX-LM runtime, and a QLoRA candidate. `--training-runtime mlx-lm` can pin the
runtime explicitly, but the documented MPS inference rule selects it when the
flag is omitted. `--prefer-method qlora` remains an optional tie-breaker because
the exact policy makes every other MoE method unsupported. The policy also
rejects shared experts and all placements except `single`. Every accepted row
remains conditional and pilot-required.

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

The command profiles the source, constructs the supplied model and hardware
facts, evaluates one model-policy decision, enumerates the 12 candidates, and
writes one `aptus.training-plan.v5` document. It binds the deterministic
model-policy snapshot digest. Without an inspection receipt,
the decision source is `user-attested`. Parent directories are created. An
existing plan output file is replaced. No bundle or archive is created.

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

The plan is rehydrated through the exact v5 domain contract. A saved v4, v3,
v2, or schema-less plan fails with `Replan required`. A coherent v5 plan whose
policy decision or snapshot digest differs from the current host registry fails
the same way; malformed or tampered v5 policy state is an ordinary validation
error. Aptus leaves the source plan unchanged and creates no bundle. Recreate
the plan deterministically from its preserved facts. Do not relabel it as v5.

The default archive is the bundle path with its suffix replaced by `.zip`. The
archive must be outside the bundle. Bundle and archive publication are
no-clobber. On success the command prints the bundle path, archive path, and
static validation report. The persisted candidate runtime contract controls
compilation. `compile` does not accept a runtime override.

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

The installed Aptus CLI performs host-side policy currency checks against the
current registry. By contrast, invoking a generated `validate.py` directly in a
package-free transferred bundle verifies the integrity of its frozen snapshot
and parity of the saved decision; it cannot determine host policy currency or
know whether the installed policy on some other host has advanced.

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

For CUDA, `pilot` performs two fresh-process phases and checkpoint continuation.
For MLX-LM, `pilot` performs one uninterrupted two-update adapter run, followed
by fresh-process adapter reload and one-to-four-token generation. A passing MLX
pilot permits confirmed uninterrupted full-duration adapter training. It does
not enable resume.

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

Job records use `aptus.job-record.v1`. Schema-less legacy records migrate with
durable authorization cleared. Corrupt, symlinked, or unsupported records move
to private quarantine with a reason receipt instead of blocking healthy jobs.

## `aptus doctor`

```bash
aptus doctor [--state-dir .aptus-state] [--output REPORT.json]
```

The doctor is read-only. It inventories likely exact Python interpreters,
runtime import and exact-pin compatibility results, configured runtime keys,
and bounded local-state counts. On Apple Silicon its preferred runtime is
`mlx-lm`. Only a compatible interpreter yields
`status: ready` and exit `0`. Otherwise the report uses `status:
action-required`, prints exact next steps, and exits `2`. The report always sets
`installation_performed: false`. Reports omit interpreter paths, include only
short path fingerprints, and replace the state path with `$HOME` or a
fingerprint. Runtime error bodies are reduced to presence flags. With `--output`,
the ordinary JSON writer creates or replaces the named report.

## `aptus diagnostics`

```bash
aptus diagnostics [--state-dir .aptus-state] --output aptus-diagnostics.zip
```

The command creates a new, no-clobber mode-0600 ZIP. It contains `README.txt`,
`diagnostics.json`, and a SHA-256 `manifest.json`. The diagnostic payload covers
the Aptus and Python versions, bounded host and Apple platform facts, disk
capacity, runtime doctor results, job state/action counts, project and revision
counts, and quarantine counts.

The archive excludes logs, dataset and model content, project names,
environment values, and unredacted home paths. Review `diagnostics.json` before
sharing it. An existing output path is an error and is never overwritten.

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
CUDA reports one `mps` discovery record for shared unified memory. MLX-LM LoRA
and QLoRA bundles can run managed validation, uninterrupted pilot, and confirmed
full-duration adapter training when a compatible interpreter is configured.
MLX resume remains unsupported. Other unmeasurable hosts return
`status: unavailable`, `manual_facts_supported: true`, and exit `2`.

## `aptus inspect model`

```bash
aptus inspect model --model-id REPOSITORY \
  --revision REVISION [--timeout 10]
```

The timeout must be greater than zero and no more than 30 seconds. Inspection
makes bounded Hugging Face config and metadata requests. It returns status
`ok`, `unavailable`, or `unsupported`. Only `ok` exits successfully. Provider
metadata never decides parameter count or training permission. A successful
result contains an `aptus.model-inspection-receipt.v1` with separate
compatibility-subject and observed-planning-facts digests. Either save the whole
inspection output or extract its `inspection_receipt` object for a later
`--inspection-receipt` planning argument.

## `aptus serve`

```bash
aptus serve [--host 127.0.0.1] [--port 8787] \
  [--state-dir .aptus-state] [--web-dist DIST] \
  [--allow-non-loopback]
```

The command requires the `server` optional dependency group. The packaged
workbench is used unless `--web-dist` names an explicit valid build. Binding to
anything other than `127.0.0.1`, `localhost`, or `::1` is rejected unless
`--allow-non-loopback` is present.

Every launch generates a new random session token. Before Uvicorn starts, the
CLI prints these values to standard error:

```text
Aptus workbench: http://127.0.0.1:8787/?aptus_session_token=TOKEN
Aptus API bearer token: TOKEN
```

Open the printed workbench URL. The first valid public GET exchanges the query
token for an HttpOnly, SameSite Strict cookie, then returns `303` to the same
path without `aptus_session_token`. Subsequent browser requests use the cookie.
API clients can instead send `Authorization: Bearer TOKEN`.

Only `GET /api/v1/health`, `GET /health`, and static workbench assets are public.
All other API routes, `/docs`, `/redoc`, and `/openapi.json` require the cookie
or bearer token. The CLI runs Uvicorn with access logging disabled so the query
handoff is not written to normal request logs. Treat the printed URL and token
as credentials despite that protection.

`--allow-non-loopback` prints a warning and allows all Host headers. Session
authentication remains active, but Aptus still serves plain HTTP. A network
observer can steal the cookie or bearer token. Use an approved TLS and network
boundary. The flag adds no tenant isolation, bundle-root policy, or worker
isolation.

## Exit status

| Status | Meaning |
| ---: | --- |
| `0` | Command succeeded, report was not invalid, or managed job completed |
| `1` | Validation report is `invalid`, or managed job ended outside `completed` |
| `2` | Argument, input, inspection, JSON, filesystem, command error, or doctor action required |
| `130` | The CLI handled an interrupt by requesting job cancellation |

Argument-parser usage errors also exit `2`.

## Related documentation

- [Configuration defaults](configuration-defaults.md)
- [API reference](api.md)
- [Plan schema](plan-schema.md)
- [Bundle manifest](bundle-manifest.md)
- [Run states](run-states.md)
- [Validation states](validation-states.md)
