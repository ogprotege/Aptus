# Security Policy

## Supported status

Aptus v0.2 is an engineering preview. It has not passed the required CUDA pilot
and release evidence gates. Report security issues privately to the repository
owner. Do not include credentials, private datasets, or model tokens in a public
issue.

## Trust model

Aptus is a single-user, local-host tool. The API can read local datasets, write
bundles and job state, fetch bounded model metadata, and start Python training
processes. It has no authentication, tenant isolation, CSRF protection, or
remote-user authorization.

Keep `aptus serve` on `127.0.0.1`. `--allow-non-loopback` is an explicit
acknowledgment, not a security control. A non-loopback deployment requires an
external authenticated boundary, origin controls, TLS, and filesystem
isolation.

## Data copies

Compilation intentionally creates cleartext copies of the source data:

- `data/dataset.*` is the copied source.
- `data/training.jsonl` contains every canonical training row.
- `data/pilot-sample.jsonl` contains the bounded pilot pressure set.
- The bundle ZIP contains those files again.

Validation and training may also place tokenizer, model, checkpoint, log,
metrics, and final-export material on disk. Generated runtime outputs are not
part of the compiler manifest, but they remain sensitive files. Protect the
bundle directory, archive, `.aptus-state`, cache directories, and backups.

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
At train admission, Aptus deeply verifies bundle, plan, environment, pilot
metrics, pilot checkpoint and export artifacts, current CUDA identity and free
VRAM, free host RAM, and free disk. A status page may show cached evidence. The
admission transaction is authoritative.

Cancellation targets the recorded process group on POSIX systems. The service
records `cancelling` while termination is in progress. A result is not promoted
to success merely because a process disappeared.

Full-training resume is disabled. Aptus does not accept an arbitrary checkpoint
path because v0.2 cannot yet prove complete model, optimizer, scheduler, scaler,
RNG, environment, and plan continuity for a general full run.

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
