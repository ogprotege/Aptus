# Security Boundaries

> **Status:** Active | **Authority:** Normative security architecture | **Applies to:** Aptus 0.2 | **Audience:** Operators, integrators, and security reviewers | **Last reviewed:** 2026-08-14 | **Review by:** 2026-11-01 or after a trust-boundary change

## Authenticated local service boundary

The FastAPI service can read data, write files, fetch model metadata, and launch
training processes. `aptus serve` creates a new random token for every launch
and passes it to the application. Only health responses and static workbench
assets are public. Every other API route, OpenAPI schema, and interactive API
page requires either the session cookie or an `Authorization: Bearer TOKEN`
header.

The CLI prints the workbench origin without a query token and prints the same
value as an API bearer token. A valid GET to a public workbench path may still
exchange an operator-supplied `aptus_session_token` query for an HttpOnly,
SameSite Strict cookie. The service immediately returns `303` to the same path
without the token query. Invalid handoff values return `403`. Treat the printed
token as a credential. Do not put it in the URL unless you accept browser
history exposure.

The CLI disables Uvicorn access logs. These controls do not make the token safe
to disclose through terminal capture, process supervision, or another observer.

The default service accepts only loopback Host headers, including `localhost`,
`127.0.0.1`, and `[::1]`. Non-loopback binding is rejected unless
`--allow-non-loopback` is explicit. That mode still requires the token, but the
server uses plain HTTP. Anyone who can observe the network can steal the token.
Use an approved TLS and network boundary. The flag does not add tenant
isolation, filesystem scoping, worker isolation, or a remote-user policy.

`create_app()` requires `session_token` unless the caller passes
`allow_unauthenticated=True`. That programmatic opt-out has no application
authentication and must remain behind a suitable local or external boundary.

## Native desktop boundary

Aptus for Mac starts a separate desktop entrypoint on an ephemeral loopback
port. The native host creates a random per-launch token and installs it as an
HttpOnly, SameSite Strict cookie before WebKit loads the workbench. Every
protected API route requires that cookie. Static assets and health remain
public. The token is absent from the URL, readiness file, JavaScript bridge,
state files, and logs. Desktop responses add a
same-origin content security policy, deny framing, suppress referrers, and
disable MIME sniffing.

The desktop boundary does not turn the API into a secure remote or multi-user
service. WebKit accepts only the exact loopback session origin. External links
open outside the app. Unlike `aptus serve`, the desktop host installs the cookie
before its first request and never places the token in a URL.

The desktop sidecar keeps every runtime job bound to the compiled runtime
contract. MLX-LM jobs use a probed external interpreter. CUDA bundles remain a
target-host handoff in the Mac UI, and a manual CUDA profile never becomes
evidence that the Mac has CUDA.

The native runtime-configuration client accepts only the exact authenticated
loopback origin. It sends the same session cookie through an ephemeral URL
session, refuses redirects, and bounds the response. The selected interpreter
must be an executable regular path whose runtime probe passes.

## Filesystem boundary

Paths are resolved before access. Compiler output is no-clobber and atomic.
Bundle integrity covers compiler-created files. Runtime output has separate
attestations because checkpoints and exports are intentionally mutable while a
job runs.

The embedded model-policy snapshot is one of those compiler-created files. Its
canonical bytes, plan binding, manifest binding, and manifested digest establish
portable integrity. They do not establish that the frozen policy remains
current on an installed host.

The compiler makes cleartext dataset copies and a ZIP. Runtime adds logs,
metrics, checkpoints, model or adapter files, tokenizer data, and caches. OS
backup and sync software can duplicate all of them.

The selected runtime path is stored in `runtime-config.json` with mode 0600.
The state loader rejects a symlink or non-regular configuration file. Protect
the referenced environment from replacement by other users or processes.

State roots, plans, project manifests, immutable revisions, current pointers,
runtime configuration, job records, and quarantine receipts use mode-0700
directories and mode-0600 JSON files on POSIX systems. Loaders reject symlinks.
Corrupt and unsupported project or job state is moved aside with a private
reason receipt. Quarantine is recoverable containment, not deletion.

Diagnostic archives are privacy bounded. They include host and runtime facts,
disk capacity, and state counts. They exclude logs, source data, model content,
project names, environment values, and unredacted home paths. The output is
no-clobber and mode 0600.

## Model-provider boundary

Model inspection is a bounded metadata fetch from a declared repository and
revision. The model ID must match the provider repository identifier used by
planning. Fetches disable HTTP proxies, stay on `https://huggingface.co`, and
refuse a response that leaves that origin. Returned fields are untrusted
provider declarations. They do not prove the license, permission to train,
absence of remote code, artifact safety, or compatibility with the pinned
runtime.

Runtime model loading is bound to the pinned revision. Operators must handle
credentials through the training stack's normal secure mechanisms. Do not place
tokens in plan JSON, source data, generated code, or job requests.

## Model-policy boundary

The installed model-compatibility registry is the authority for host policy
currency. Package-free validation has only the bundle's embedded frozen
snapshot and can establish integrity and decision parity, not currentness. An
internally coherent v5 plan whose decision or digest differs from the installed
registry requires replanning before host-managed admission.

Managed submission records the current approved snapshot digest. The parent
rechecks it before releasing the launch permit and passes it to generated
entrypoints as `APTUS_AUTHORIZED_MODEL_POLICY_SNAPSHOT_SHA256`. Direct portable
execution has no such host authorization and must not be represented as current
host approval. Pilot authorization, worker launch, and the completion
verification and promotion transaction repeat the installed-registry check, so
historical evidence cannot cross a policy change silently.

## Local inference boundary

LM Studio and oMLX integrations accept one explicit HTTP loopback origin with
an explicit port. They disable proxies and redirects, bound request and response
sizes, and use finite timeouts. They do not scan arbitrary ports, accept LAN
hosts, or become training runtimes. Responses are untrusted inference output.

## Code-execution boundary

Generated `train.py`, `preflight.py`, `validate.py`, `run.py`, and MLX
`reload.py` and `eval.py` execute Python in the bundle environment. They also import the
manifested `plan_contract.py`, `policy_snapshot.py`, and `runtime_lease.py`
helpers. Package installation and model loading can execute third-party code.
Review direct pins, resolved dependencies, model settings, and the bundle before
execution.

`requirements.txt` is an exact direct constraint set, not a transitive lock.
MLX-LM and PyTorch MPS runtime selection can launch an external Python
interpreter. Selecting it grants that environment the same dataset, bundle, and
artifact access as the Aptus job. Review and control the environment first.

## Concurrency boundary

Managed jobs and POSIX portable entrypoints use one host-global Aptus lease for
the same local user. It does not reserve CUDA or unified memory against unrelated software.
Operators remain responsible for host scheduling outside Aptus. Direct portable
child execution is fail-closed on Windows in v0.2.

## Admission boundary

A pilot report is historical evidence. Train admission deeply verifies current
runtime-specific bindings and capacity under the execution lease. Current free
resources can invalidate an earlier pass. Authorization is not delegated to UI
state. A changed current model-policy snapshot also invalidates authorization
and requires replanning even when the old pilot remains internally coherent.

CUDA admission verifies environment, hardware, checkpoint, export, VRAM, host
RAM, and disk evidence. MLX admission verifies its immutable uninterrupted
pilot, then requires current available unified memory above measured pilot peak
plus reserve and enough disk. MLX fresh-process adapter reload does not create a
training-resume capability.

## Completion boundary

A child process cannot declare the job successful. The parent verifies pending
metrics and the final structural safetensors file tree, rechecks current host
policy, then promotes the report. A policy change leaves pending evidence
unpromoted.
Historical polling retains that completion-time attestation but does not rehash
all large files continuously.

Structural export verification does not establish semantic model behavior,
benchmark quality, safety, or deployment fitness.

## Unsupported boundaries

V0.2 has no secure multi-user service, cloud credential broker, provider
provisioner, MCP authorization policy, full-run resume contract, PyTorch MPS
compiler, or evaluation policy engine. MLX-LM train authorization exists only
as a fresh, lease-held admission transaction. It is never durable project state.

## Related documentation

- [Security policy](../../SECURITY.md)
- [Data and identity flow](data-and-identity-flow.md)
- [Bundle manifest](../reference/bundle-manifest.md)
- [Model-policy snapshot](../reference/model-policy-snapshot.md)
- [Operator checklist](../operations/operator-checklist.md)
- [macOS desktop host](macos-desktop.md)
