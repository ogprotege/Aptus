# CLI Reference

Run `aptus COMMAND --help` for the exact installed options.

## `aptus profile`

Profiles a local dataset.

```bash
aptus profile --dataset DATASET [--sample-limit 512] \
  [--sequence-length TOKENS] [--output PROFILE.json]
```

The sample limit bounds profiling statistics. Compilation still validates and
writes every supported source-schema row.

## `aptus spec-plan`

Writes a v2 plan without compiling it. Required arguments cover model identity
and architecture, license and permission, dataset, GPU and host resources,
objective, sequence length, and output path.

Important switches include:

```text
--confirm-training-allowed
--backend {cuda,rocm,mps,cpu}
--bf16 --four-bit --eight-bit
--prefer-method {full,lora,int8-lora,qlora}
--objective {quality,memory,speed}
--packing
```

Known backend values do not imply execution support. V0.2 executes CUDA only.
Packing is fail-closed.

## `aptus plan` and `aptus build`

These compatibility commands create a plan, compile a bundle, validate it
statically, and create an archive. `--plan-output` optionally preserves the
standalone plan.

```bash
aptus build FACT_ARGUMENTS --output BUNDLE [--plan-output PLAN.json]
```

## `aptus compile`

```bash
aptus compile --plan PLAN.json --output BUNDLE [--archive BUNDLE.zip]
```

Output and archive are no-clobber. The archive must be outside the bundle.

## `aptus validate`

```bash
aptus validate BUNDLE \
  [--level {contract,static,dependency,model-data,measured-preflight,pilot}] \
  [--run] [--state-dir STATE_DIR]
```

Contract and static checks run directly. For a runtime level, `--run` submits a
managed job and waits for its terminal state.

## `aptus run`

```bash
aptus run BUNDLE \
  [--action {dependency,model-data,preflight,pilot,train}] \
  [--confirm-full-train] [--state-dir STATE_DIR]
```

Run actions in the displayed order. Confirmation is valid only for `train` and
is required. Full-run resume has no CLI option.

## `aptus jobs`

```bash
aptus jobs [--state-dir STATE_DIR] [--id JOB_ID]
```

Without an ID, prints persisted jobs. With an ID, prints the reconciled record.

## `aptus hardware` and `aptus inspect hardware`

Inspect local hardware facts. CUDA hosts report each visible device and current
capacity. On Darwin arm64 without CUDA, discovery reports one `mps` device for
the measured shared unified-memory pool. That record does not enable MPS or MLX
execution, and current availability remains unknown when macOS does not expose
it. Other unmeasurable hosts return status `unavailable` with manual-fact
support. V0.2 execution remains CUDA-only.

## `aptus inspect model`

```bash
aptus inspect model --model-id REPOSITORY \
  --revision IMMUTABLE_COMMIT [--timeout 10]
```

Returns bounded provider-declared facts and warnings. It does not decide training
permission.

## `aptus serve`

```bash
aptus serve [--host 127.0.0.1] [--port 8787] \
  [--state-dir .aptus-state] [--web-dist DIST]
```

Non-loopback binding is blocked unless `--allow-non-loopback` is supplied. That
flag is only an acknowledgment. It does not add authentication.

## Exit status

- `0`: command or managed job completed successfully.
- `1`: validation invalid or managed job did not complete successfully.
- `2`: input, inspection, filesystem, or command error.
- `130`: interrupted managed job cancellation path.
