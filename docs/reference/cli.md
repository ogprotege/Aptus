# CLI Reference

| Metadata | Value |
| --- | --- |
| Status | Active |
| Audience | Local operators, developers, and automation authors |
| Authority | Normative reference for the Aptus v0.2 command-line contract |
| Last reviewed | 2026-08-05 |
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
| `spec-plan` | Write a standalone v6 plan | Objective `memory` | Plan JSON output file |
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
| `eval-contract` | Bind a gold JSONL into `aptus.evaluation-contract.v1` | `exact_match` | Optional contract JSON and optional presentation-only plan copy |
| `eval` | Score predictions against a bound contract | Write result JSON | Optional result JSON; does not start a training job |
| `serve` | Serve an authenticated API and packaged workbench | `127.0.0.1:8787` | Per-launch workbench handoff, bearer token, state, and job records |

## Machine-readable parser contract

This JSON document is checked against the live `argparse` tree. `null` is the
parser default when an argument is omitted, including required arguments that
must be supplied before parsing can succeed. Angle brackets identify positional
arguments and subcommand selectors. The `planning-facts` group is included by
`spec-plan`, `plan`, and `build`; it is expanded before comparison. This block
binds command and argument names, choices, and non-suppressed defaults. The
installed help and reviewed prose remain authoritative for requiredness, value
types, validation rules, side effects, and operational meaning.

<!-- aptus-cli-parser-contract:start -->
```json
{
  "schema_version": "aptus.cli-parser-contract.v1",
  "argument_groups": {
    "planning-facts": {
      "--architecture": {"default": null},
      "--backend": {"choices": ["cuda", "rocm", "mps", "cpu"], "default": "cuda"},
      "--bf16": {"default": false},
      "--checkpoint-steps": {"default": 100},
      "--data-order-seed": {"default": 1000017},
      "--confirm-training-allowed": {"default": false},
      "--confirm-unreviewed-runtime": {"default": false},
      "--context-length": {"default": null},
      "--dataset": {"default": null},
      "--disk-free-gib": {"default": null},
      "--effective-batch-size": {"default": 16},
      "--eight-bit": {"default": false},
      "--epochs": {"default": 3},
      "--evaluation-fraction": {"default": 0.1},
      "--family": {"default": null},
      "--four-bit": {"default": false},
      "--free-vram-gib": {"default": null},
      "--gpu-count": {"default": null},
      "--gradient-accumulation-steps": {"default": null},
      "--hidden-size": {"default": null},
      "--host-ram-free-gib": {"default": null},
      "--host-ram-gib": {"default": null},
      "--inspection-receipt": {"default": null},
      "--intermediate-size": {"default": null},
      "--layers": {"default": null},
      "--license": {"default": null},
      "--micro-batch-size": {"default": null},
      "--model-id": {"default": null},
      "--model-type": {"default": null},
      "--moe-decoder-sparse-step": {"default": null},
      "--moe-expert-count": {"default": null},
      "--moe-expert-intermediate-size": {"default": null},
      "--moe-experts-per-token": {"default": null},
      "--moe-mlp-only-layer": {"default": []},
      "--moe-shared-expert-intermediate-size": {"default": null},
      "--objective": {"choices": ["quality", "memory", "speed"], "default": "memory"},
      "--optimizer-steps": {"default": null},
      "--packing": {"default": false},
      "--parameters-b": {"default": null},
      "--prefer-method": {"choices": ["full", "lora", "int8-lora", "qlora"], "default": null},
      "--quantization-bits": {"choices": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], "default": null},
      "--quantization-group-size": {"default": null},
      "--quantization-layout-profile": {"choices": ["qwen3-moe-4bit-group64-router-gates-8bit"], "default": null},
      "--reserve-gib": {"default": 2.0},
      "--revision": {"default": null},
      "--sample-limit": {"default": 512},
      "--sequence-length": {"default": null},
      "--split-seed": {"default": 424242},
      "--training-runtime": {"choices": ["transformers-peft-cuda", "mlx-lm", "pytorch-mps"], "default": null},
      "--training-seed": {"default": 17},
      "--vram-gib": {"default": null}
    }
  },
  "commands": {
    "aptus": {
      "<command>": {"choices": ["profile", "spec-plan", "plan", "build", "compile", "select-candidate", "validate", "run", "jobs", "doctor", "diagnostics", "serve", "hardware", "eval-contract", "eval", "inspect"], "default": null}
    },
    "aptus profile": {
      "--dataset": {"default": null},
      "--sample-limit": {"default": 512},
      "--sequence-length": {"default": null},
      "--output": {"default": null}
    },
    "aptus spec-plan": {
      "$groups": ["planning-facts"],
      "--output": {"default": null}
    },
    "aptus plan": {
      "$groups": ["planning-facts"],
      "--output": {"default": null},
      "--plan-output": {"default": null}
    },
    "aptus build": {
      "$groups": ["planning-facts"],
      "--output": {"default": null},
      "--plan-output": {"default": null}
    },
    "aptus compile": {
      "--plan": {"default": null},
      "--output": {"default": null},
      "--archive": {"default": null}
    },
    "aptus select-candidate": {
      "--plan": {"default": null},
      "--candidate-id": {"default": null},
      "--output": {"default": null}
    },
    "aptus validate": {
      "<bundle>": {"default": null},
      "--level": {"choices": ["contract", "static", "dependency", "model-data", "measured-preflight", "pilot"], "default": "static"},
      "--run": {"default": false},
      "--state-dir": {"default": ".aptus-state"}
    },
    "aptus run": {
      "<bundle>": {"default": null},
      "--action": {"choices": ["dependency", "model-data", "preflight", "pilot", "train"], "default": "preflight"},
      "--confirm-full-train": {"default": false},
      "--state-dir": {"default": ".aptus-state"}
    },
    "aptus jobs": {
      "--state-dir": {"default": ".aptus-state"},
      "--id": {"default": null}
    },
    "aptus doctor": {
      "--state-dir": {"default": ".aptus-state"},
      "--output": {"default": null}
    },
    "aptus diagnostics": {
      "--state-dir": {"default": ".aptus-state"},
      "--output": {"default": null}
    },
    "aptus serve": {
      "--host": {"default": "127.0.0.1"},
      "--port": {"default": 8787},
      "--state-dir": {"default": ".aptus-state"},
      "--web-dist": {"default": null},
      "--allow-non-loopback": {"default": false}
    },
    "aptus hardware": {},
    "aptus eval-contract": {
      "--dataset": {"default": null},
      "--claim": {"default": null},
      "--threshold": {"default": null},
      "--metric": {"choices": ["exact_match"], "default": "exact_match"},
      "--gold-field": {"choices": ["completion", "output", "gold"], "default": "completion"},
      "--id-field": {"default": "id"},
      "--casefold": {"default": false},
      "--plan-id": {"default": null},
      "--candidate-id": {"default": null},
      "--job-id": {"default": null},
      "--export-digest": {"default": null},
      "--export-kind": {"choices": ["adapter", "final-export"], "default": null},
      "--output": {"default": null},
      "--attach-plan": {"default": null},
      "--plan-output": {"default": null}
    },
    "aptus eval": {
      "--contract": {"default": null},
      "--gold": {"default": null},
      "--predictions": {"default": null},
      "--export-digest": {"default": null},
      "--output": {"default": null}
    },
    "aptus inspect": {
      "<inspect_command>": {"choices": ["hardware", "model"], "default": null}
    },
    "aptus inspect hardware": {},
    "aptus inspect model": {
      "--model-id": {"default": null},
      "--revision": {"default": null},
      "--timeout": {"default": 10.0}
    }
  }
}
```
<!-- aptus-cli-parser-contract:end -->

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
| `--model-type TYPE` | No | `null` | Exact provider model type; required by registered policy matches, including dense Qwen2 and Qwen3 MoE |
| `--architecture CLASS` | No | `null` | Exact provider architecture class; required by registered policy matches, including dense Qwen2 and Qwen3 MoE |
| `--quantization-bits BITS` | No | `null` | Pinned checkpoint precision from 1 through 16 bits |
| `--quantization-group-size INTEGER` | No | `null` | Positive default group size for a uniform layout with no module overrides; requires `--quantization-bits` and excludes a named layout profile |
| `--quantization-layout-profile PROFILE` | No | `null` | Exact reviewed provider map; the first MoE row requires `qwen3-moe-4bit-group64-router-gates-8bit` |
| `--hidden-size INTEGER` | Yes | None | Positive hidden width |
| `--intermediate-size INTEGER` | No | `null` | Positive MLP width; required for MLP adapter targets. Aptus does not invent `4 * hidden_size` |
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
| `--confirm-unreviewed-runtime` | No | False | Attest an unreviewed Qwen2 MLX layer count (not Path Alpha). Required to plan 28-layer 4-bit dense Qwen2 such as 7B. Does not mark the path reviewed. |
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
| `--free-vram-gib NUMBER` | No | `null` | Current free memory. Omitted CUDA free memory is infeasible; Aptus will not treat total as free |
| `--bf16` | No | False | Declares BF16 capability |
| `--four-bit` | No | False | Declares four-bit capability |
| `--eight-bit` | No | False | Declares eight-bit capability |
| `--host-ram-gib NUMBER` | Yes | None | Positive total host RAM |
| `--host-ram-free-gib NUMBER` | No | `null` | Current free host RAM. Omitted free host RAM is infeasible; Apple Silicon uses this as unified headroom |
| `--reserve-gib NUMBER` | No | `2.0` | Non-negative reserve subtracted from each device. Raised to at least `8.0` when `--backend mps` is selected |
| `--disk-free-gib NUMBER` | No | `null` | Current free disk. Omitted free disk is infeasible; Aptus will not assume staging space |

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

The reviewed dense Qwen2 footprint uses `--model-type qwen2`,
`--architecture Qwen2ForCausalLM`, 24 layers, `--quantization-bits 4`, and
`--quantization-group-size 64`. The generic group-size flag emits an explicit
uniform layout with an empty override list; it is mutually exclusive with the
named Qwen3 MoE mixed-layout profile.

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
writes one `aptus.training-plan.v6` document. It binds the deterministic
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

The plan is rehydrated through the exact v6 domain contract. A saved v5, v4,
v3, v2, or schema-less plan fails with `Replan required`. A coherent v6 plan whose
policy decision or snapshot digest differs from the current host registry fails
the same way; malformed or tampered v5 policy state is an ordinary validation
error. Aptus leaves the source plan unchanged and creates no bundle. Recreate
the plan deterministically from its preserved facts. Do not relabel it as v6.

The default archive is the bundle path with its suffix replaced by `.zip`. The
archive must be outside the bundle. Bundle and archive publication are
no-clobber. On success the command prints the bundle path, archive path, and
static validation report. The persisted candidate runtime contract controls
compilation. `compile` does not accept a runtime override.

Compilation copies cleartext data and writes a mutable
`validation-report.json` to the directory. The deterministic ZIP excludes that
report and all runtime-only outputs.

## `aptus select-candidate`

```bash
aptus select-candidate --plan PLAN.json \
  --candidate-id cand_0123456789abcdef0123 --output SELECTED.json
```

The command validates the complete source plan and candidate, rejects stale,
mutated, rejected, unknown, or already-selected identities, and writes a new
no-clobber v6 plan with a new `plan_id`. Policy, inspection, evidence, and all
planning facts remain unchanged. Compilation then uses that explicitly selected
complete candidate; generated plan and trainer files are never edited.

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

The timeout must be greater than zero and no more than 30 seconds. The model ID
must be a provider repository identifier, not a URL or local path. Inspection
makes bounded Hugging Face config and metadata requests, disables HTTP proxies,
and follows redirects only while the origin remains `https://huggingface.co`.
It returns status `ok`, `unavailable`, or `unsupported`. Only `ok` exits
successfully. Provider metadata never decides parameter count or training
permission. A successful
result contains an `aptus.model-inspection-receipt.v1` with separate
compatibility-subject and observed-planning-facts digests. Either save the whole
inspection output or extract its `inspection_receipt` object for a later
`--inspection-receipt` planning argument.

## `aptus eval-contract`

```bash
aptus eval-contract --dataset GOLD.jsonl --claim TEXT --threshold 1.0 \
  [--metric exact_match] [--gold-field completion] [--id-field id] \
  [--casefold] [--plan-id PLAN_ID] [--candidate-id CANDIDATE_ID] \
  [--job-id JOB_ID] [--export-digest HEX] [--export-kind {adapter,final-export}] \
  [--output CONTRACT.json] [--attach-plan PLAN.json --plan-output PLAN-WITH-EVAL.json]
```

This writes `aptus.evaluation-contract.v1`. The gold file digest, row count,
metric implementation, and threshold are the binding. A local gold path is an
operator hint and is not contract identity. `--attach-plan` copies a persisted
plan and adds a presentation-only `evaluation_contract` field; `plan_id` does
not change. Contract and attached-plan outputs refuse to overwrite existing
files. Training finished is not an evaluation pass.

## `aptus eval`

```bash
aptus eval --contract CONTRACT.json --gold GOLD.jsonl \
  --predictions PRED.jsonl [--export-digest HEX] [--output RESULT.json]
```

This scores operator-supplied predictions against the bound gold digest using
`aptus.exact-match.v1`. Aptus does not generate predictions. The command writes
`aptus.evaluation-result.v1` and exits `0` only for `pass`. `fail` and
`abstain` exit `1`. Missing or extra prediction identities, a gold-digest
mismatch, or an export-digest mismatch abstain. This is not a managed training
job and does not change validation state.

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
Aptus workbench: http://127.0.0.1:8787/
Aptus API bearer token: TOKEN
```

The printed workbench URL does not include the session token. API clients send
`Authorization: Bearer TOKEN`. Opening the printed URL loads static workbench
assets; protected API calls still need the cookie or bearer token.

An optional query handoff remains available if the operator appends
`?aptus_session_token=TOKEN`. The first valid public GET exchanges that query
for an HttpOnly, SameSite Strict cookie, then returns `303` to the same path
without `aptus_session_token`. That path stores the token in browser history.
Prefer the printed bearer token.

Only `GET /api/v1/health`, `GET /health`, and static workbench assets are public.
All other API routes, `/docs`, `/redoc`, and `/openapi.json` require the cookie
or bearer token. The CLI runs Uvicorn with access logging disabled. Treat the
printed token as a credential.

`--allow-non-loopback` prints a warning and allows all Host headers. Session
authentication remains active, but Aptus still serves plain HTTP. A network
observer can steal the cookie or bearer token. Use an approved TLS and network
boundary. The flag adds no tenant isolation, bundle-root policy, or worker
isolation.

## Exit status

| Status | Meaning |
| ---: | --- |
| `0` | Command succeeded, report was not invalid, managed job completed, or eval passed |
| `1` | Validation report is `invalid`, managed job ended outside `completed`, or eval failed/abstained |
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
