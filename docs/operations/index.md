# Operations

> **Status:** Active | **Authority:** Documentation navigation | **Applies to:** Aptus 0.2 | **Audience:** Operators and release maintainers | **Last reviewed:** 2026-08-06 | **Review by:** 2026-10-27 or when operational behavior changes

## Run a bundle

- [Operator checklist](operator-checklist.md)
- [Compile, validate, and run](../guides/compile-validate-run.md)
- [State, storage, and retention](state-storage-retention.md)
- [Recovery and resume boundary](../guides/resume-recover.md)
- [Security policy](../../SECURITY.md)

## Prepare a release

- [Release gates](release-gates.md)
- [Release evidence template](release-evidence-template.md)
- [2026-08-06 SmolLM2 CUDA LoRA single-device acceptance](evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/README.md)
- [2026-08-05 Phase 6 Qwen2 MLX-LM exact-source acceptance](evidence/2026-08-05-qwen2-mlx-lm-exact-source-refresh/README.md)
- [2026-08-05 original Phase 6 Qwen2 MLX-LM acceptance baseline](evidence/2026-08-05-qwen2-mlx-lm-acceptance/README.md)
- [2026-07-27 MLX-LM target-host acceptance](evidence/2026-07-27-mlx-lm-acceptance/README.md)
- [2026-07-27 desktop engineering acceptance](evidence/2026-07-27-desktop-release/README.md)
- [Changelog](../../CHANGELOG.md)

## Experimental host work

- [Apple Silicon fine-tuning experiment matrix](apple-silicon-pilot.md)

That Apple matrix combines current v5/v3 small-model MLX-LM QLoRA acceptance
at exact source `719255153e3fc7e38e83b5ff826d587e5e58bf80`, the original Phase 6
baseline, historical v2/v2 evidence, and proposed larger-model work. The
separate record above covers one exact 2026-08-06 CUDA LoRA single-device
acceptance and identifies the remaining CUDA target-host work. Every result is
scoped to its exact artifact, source tree,
bundle fingerprint, host, and environment; none transfers to another runtime
path or establishes repeatability, safety, quality, performance, production
readiness, or release readiness. Read the linked immutable evidence record
before treating a row as passed.

## Related documentation

- [Documentation home](../index.md)
- [Run states](../reference/run-states.md)
- [Troubleshooting](../guides/troubleshooting.md)
