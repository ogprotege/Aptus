# Security Policy

> **Status:** Active | **Authority:** Normative security policy | **Applies to:** Aptus 0.2 | **Audience:** Users, operators, and maintainers | **Last reviewed:** 2026-07-27 | **Review by:** 2026-10-27 or after a trust-boundary change

## Supported status

Aptus v0.2 is an engineering preview. Apple Silicon MLX-LM acceptance reached
`measured-run-pass` twice in a clean isolated checkout. Ten of 10 clean local
desktop engineering builds passed at implementation commit
`1038ecdd13103418ef1135e1ced634c10370a961`. That historical gate proves the
recorded source and ad-hoc-signed packages only. Pull-request CI must rebuild the
exact workflow commit. Pull requests use GitHub's synthetic merge commit. CUDA
target-host evidence and public Developer ID signing and notarization remain
open. Use GitHub's
[private vulnerability-reporting flow](https://github.com/ogprotege/Aptus/security/advisories/new)
when it is available. Otherwise contact the repository owner through an
existing private channel before sharing technical details. Do not include
credentials, private datasets, model tokens, exploit details, or unredacted
logs in a public issue.

| Version | Security fixes | Status |
|---|---|---|
| `0.2.x` and current `main` | Yes | Engineering preview |
| `0.1.x` and legacy audit sources | No | Archived or superseded |

Include the affected commit or version, impact, minimal reproduction, and the
smallest redacted evidence needed to verify the report. The maintainer target is
an acknowledgment within three business days and an initial assessment within
seven. These are response targets, not a service-level guarantee.

Keep the report private until the maintainer confirms a fix or agrees to a
disclosure date. The project will credit reporters who request attribution and
will not publish sensitive reproduction material. If active exploitation makes
continued privacy unsafe, coordinate the minimum necessary disclosure first.

## Trust model

Aptus is a single-user, local-host tool. The API can read local datasets, write
bundles and job state, fetch bounded model metadata, and start Python training
processes. `aptus serve` creates a random per-launch session token and requires
its HttpOnly cookie or bearer form on protected routes. It still has no tenant
isolation, remote-user authorization, TLS, or worker sandbox.

Keep `aptus serve` on `127.0.0.1`. `--allow-non-loopback` is an explicit
acknowledgment, not a security control. A non-loopback deployment requires an
external authenticated boundary, origin controls, TLS, and filesystem
isolation.

Aptus for Mac starts a distinct service on an ephemeral loopback port. Every
protected route requires a random per-launch HttpOnly, SameSite Strict cookie
installed by the native host. WebKit stays on the exact session origin. This
protects the desktop sidecar from unrelated local web pages, but it does not
create a remote or multi-user service boundary. Desktop responses enforce a
same-origin content security policy and deny framing. The sidecar can run an
eligible MLX-LM bundle through the configured external interpreter. CUDA
remains an external target-host path.

## Data copies

Compilation intentionally creates cleartext copies of the source data:

- `data/dataset.*` is the copied source.
- `data/training.jsonl` contains every canonical training row.
- `data/pilot-sample.jsonl` contains the bounded pilot pressure set.
- MLX bundles also contain disjoint, microbatch-padded `data/mlx/train.jsonl`
  and `data/mlx/valid.jsonl` files plus their split contract.
- The bundle ZIP contains those files again.

Validation and training may also place tokenizer, model, CUDA checkpoint, MLX
adapter or weight snapshot, log, metrics, and final-export material on disk.
Generated runtime outputs are not part of the compiler manifest, but they
remain sensitive files. Protect the bundle directory, archive, `.aptus-state`,
cache directories, and backups. The selected state root and its project, plan,
runtime-configuration, job, current-pointer, and quarantine records use
user-private directories and mode-0600 JSON files on POSIX systems. State
readers reject symlink records. Unreadable or unsupported job and project
records move into private quarantine with a reason receipt, so one corrupt
record does not hide healthy state.

`aptus diagnostics` creates a mode-0600 ZIP with bounded host facts, runtime
probe results, disk capacity, and state counts. It excludes logs, dataset and
model content, project names, environment values, and unredacted home paths.
Review `diagnostics.json` before sharing the archive.

## Input and path controls

- Local dataset and output paths are resolved before access.
- Compilation refuses a non-empty output directory and refuses an existing
  archive target.
- Bundle integrity binds compiler-created files and source data by hash.
- Model IDs use repository syntax and revisions must be immutable identifiers.
- Local model paths are outside the current model contract.
- Provider metadata is untrusted input. Inspection is bounded and does not
  establish license terms, training permission, or model safety.
- Generated Python uses structured plan data. Review a bundle before running it.

## Execution controls

Managed runtime actions use one Aptus job at a time across local state roots for
the same user. A host-global lease and per-state record locks enforce that
contract. This coordinates Aptus instances only. It does not reserve the GPU
against unrelated processes.

Full training requires explicit confirmation and a current pilot attestation.
At train admission, Aptus deeply verifies the bundle, plan, environment, pilot
metrics, runtime-specific pilot artifacts, current hardware identity, memory
headroom, and free disk. CUDA admission verifies checkpoint continuation,
export evidence, CUDA identity, free VRAM, and free host RAM. MLX admission
verifies the exact target census, changed adapter weights, positive measured
peak and delta, immutable artifacts, live unified-memory headroom, and a fresh
adapter reload that generates one to four tokens. A status page may show cached
evidence. The admission transaction is authoritative.

Cancellation targets the recorded process group on POSIX systems. The service
records `cancelling` while termination is in progress. A result is not promoted
to success merely because a process disappeared.

The native Mac host separately owns its sidecar process tree. Quit and restart
wait for typed shutdown success. Shutdown tracks PID plus process-start
identity, expands late descendants, distinguishes zombies and PID reuse, and
records its signal attempts. A survivor retains controller, path, and session
ownership, blocks replacement startup, and causes the application to refuse
termination until an explicit shutdown retry succeeds.

Full-training resume is disabled. CUDA does not accept an arbitrary checkpoint
path because v0.2 cannot yet prove complete model, optimizer, scheduler, scaler,
RNG, environment, plan, and data continuity for a general full run. Every MLX
resume argument is rejected. Periodic MLX files are weight snapshots, not
resumable checkpoints, and an MLX full run starts uninterrupted from the pinned
base model after `pilot-pass`.

## Artifact verification boundary

A successful full run writes to a unique `runs/run_*` directory. The parent
process verifies finite metrics, plan and candidate bindings, per-rank evidence,
and a structural safetensors export file tree. It then promotes the report to
`measured-run-pass`. This is structural integrity evidence, not an evaluation of
model quality or behavior.

Historical job reads use a cheap presence check and retain the completion-time
attestation. They do not continuously rehash large artifacts.

## Dependency boundary

Generated `requirements.txt` contains exact direct pins for the selected method.
It is not a transitive dependency lock. Install in an isolated environment,
review resolved dependencies, and retain the environment binding. Do not run
untrusted bundles or install unreviewed packages on a sensitive host.

The 2026-07-27 web audit reports zero production dependency advisories. The
full development audit reports four high-severity transitive advisories in the
OpenAPI generator chain: `@redocly/openapi-core`, `js-yaml`, `minimatch`, and
`brace-expansion`. That generator processes the trusted checked-in OpenAPI
document, but the advisories remain open release-engineering debt. Do not feed
it untrusted schemas.

## Related documentation

- [Security boundaries](docs/architecture/security-boundaries.md)
- [Operator checklist](docs/operations/operator-checklist.md)
- [Bundle manifest](docs/reference/bundle-manifest.md)
- [Reviewed corpus contract](docs/reference/reviewed-corpus-contract.md)
