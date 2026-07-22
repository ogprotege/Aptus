# Security Boundaries

> **Status:** Active | **Authority:** Normative security architecture | **Applies to:** Aptus 0.2 | **Audience:** Operators, integrators, and security reviewers | **Last reviewed:** 2026-07-22 | **Review by:** 2026-10-22 or after a trust-boundary change

## Local trusted-user boundary

The FastAPI service can read data, write files, fetch model metadata, and launch
training processes. It has no authentication, tenant isolation, or remote-user
policy. Serve it on loopback. Treat anyone who can call the API as able to act
with the Aptus process user's filesystem and compute permissions.

The default service accepts only loopback Host headers, including `localhost`,
`127.0.0.1`, and `[::1]`. This blocks the common DNS-rebinding path into an
unauthenticated local API. Explicit non-loopback mode relaxes that check and
requires an external authenticated boundary.

## Filesystem boundary

Paths are resolved before access. Compiler output is no-clobber and atomic.
Bundle integrity covers compiler-created files. Runtime output has separate
attestations because checkpoints and exports are intentionally mutable while a
job runs.

The compiler makes cleartext dataset copies and a ZIP. Runtime adds logs,
metrics, checkpoints, model or adapter files, tokenizer data, and caches. OS
backup and sync software can duplicate all of them.

## Model-provider boundary

Model inspection is a bounded metadata fetch from a declared repository and
revision. Returned fields are untrusted provider declarations. They do not prove
the license, permission to train, absence of remote code, artifact safety, or
compatibility with the pinned runtime.

Runtime model loading is bound to the pinned revision. Operators must handle
credentials through the training stack's normal secure mechanisms. Do not place
tokens in plan JSON, source data, generated code, or job requests.

## Code-execution boundary

Generated `train.py`, `preflight.py`, `validate.py`, and `run.py` execute Python
in the bundle environment. Package installation and model loading can execute
third-party code. Review direct pins, resolved dependencies, model settings, and
the bundle before execution.

`requirements.txt` is an exact direct constraint set, not a transitive lock.

## Concurrency boundary

Managed jobs and POSIX portable entrypoints use one host-global Aptus lease for
the same local user. It does not reserve CUDA against unrelated software.
Operators remain responsible for host scheduling outside Aptus. Direct portable
child execution is fail-closed on Windows in v0.2.

## Admission boundary

A pilot report is historical evidence. Train admission deeply verifies current
bindings and capacity under the execution lease. Current free resources can
invalidate an earlier pass. Authorization is not delegated to UI state.

## Completion boundary

A child process cannot declare the job successful. The parent verifies pending
metrics and the final structural safetensors file tree, then promotes the report.
Historical polling retains that completion-time attestation but does not rehash
all large files continuously.

Structural export verification does not establish semantic model behavior,
benchmark quality, safety, or deployment fitness.

## Unsupported boundaries

V0.2 has no secure multi-user service, cloud credential broker, provider
provisioner, MCP authorization policy, full-run resume contract, or evaluation
policy engine. Those are future designs, not hidden current features.

## Related documentation

- [Security policy](../../SECURITY.md)
- [Data and identity flow](data-and-identity-flow.md)
- [Bundle manifest](../reference/bundle-manifest.md)
- [Operator checklist](../operations/operator-checklist.md)
