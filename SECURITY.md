# Security Policy

> **Status:** Active | **Authority:** Normative security policy | **Applies to:** Aptus 0.2 | **Audience:** Users, operators, and maintainers | **Last reviewed:** 2026-07-22 | **Review by:** 2026-10-22 or after a trust-boundary change

## Supported status

Aptus v0.2 is an engineering preview. It has not passed the required target-host
CUDA or MLX release evidence gates. Use GitHub's
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
processes. The ordinary `aptus serve` interface has no authentication, tenant
isolation, CSRF protection, or remote-user authorization.

Keep `aptus serve` on `127.0.0.1`. `--allow-non-loopback` is an explicit
acknowledgment, not a security control. A non-loopback deployment requires an
external authenticated boundary, origin controls, TLS, and filesystem
isolation.

Aptus for Mac starts a distinct service on an ephemeral loopback port. Every
route requires a random per-launch HttpOnly, SameSite Strict cookie installed
by the native host. WebKit stays on the exact session origin. This protects the
desktop sidecar from unrelated local web pages, but it does not create a remote
or multi-user service boundary. Desktop responses enforce a same-origin content
security policy and deny framing. The desktop sidecar also rejects runtime
validation and job submission. CUDA and MLX actions must run outside the
sidecar on the selected target host.

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
cache directories, and backups.

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

## Related documentation

- [Security boundaries](docs/architecture/security-boundaries.md)
- [Operator checklist](docs/operations/operator-checklist.md)
- [Bundle manifest](docs/reference/bundle-manifest.md)
- [Reviewed corpus contract](docs/reference/reviewed-corpus-contract.md)
