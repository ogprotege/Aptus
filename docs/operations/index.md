# Operations

> **Status:** Active | **Authority:** Documentation navigation | **Applies to:** Aptus 0.2 | **Audience:** Operators and release maintainers | **Last reviewed:** 2026-08-05 | **Review by:** 2026-10-27 or when operational behavior changes

## Run a bundle

- [Operator checklist](operator-checklist.md)
- [Compile, validate, and run](../guides/compile-validate-run.md)
- [State, storage, and retention](state-storage-retention.md)
- [Recovery and resume boundary](../guides/resume-recover.md)
- [Security policy](../../SECURITY.md)

## Prepare a release

- [Release gates](release-gates.md)
- [Release evidence template](release-evidence-template.md)
- [2026-08-05 Phase 6 Qwen2 MLX-LM target-host acceptance](evidence/2026-08-05-qwen2-mlx-lm-acceptance/README.md)
- [2026-07-27 MLX-LM target-host acceptance](evidence/2026-07-27-mlx-lm-acceptance/README.md)
- [2026-07-27 desktop engineering acceptance](evidence/2026-07-27-desktop-release/README.md)
- [Changelog](../../CHANGELOG.md)

## Experimental host work

- [Apple Silicon fine-tuning experiment matrix](apple-silicon-pilot.md)

That matrix combines current v5/v3 small-model QLoRA acceptance at
`14ed44b52a76bb84d8d9db4f2303951aa641339b`, historical v2/v2 evidence, and
proposed larger-model and LoRA target-host work. The current result is scoped to
the exact accepted artifact and does not close CUDA or another artifact's
required runtime gates. Read the linked immutable evidence record before
treating a row as passed.

## Related documentation

- [Documentation home](../index.md)
- [Run states](../reference/run-states.md)
- [Troubleshooting](../guides/troubleshooting.md)
