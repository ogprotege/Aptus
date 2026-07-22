# Capability Matrix

This matrix describes the v0.2 compiler catalog. Every executable row still
requires static, dependency, model-data, measured-preflight, and pilot evidence
for the exact bundle and target host.

## Method and placement

| Method | Single | DDP | FSDP | Notes |
|---|---:|---:|---:|---|
| Full | Supported with BF16 | Supported with BF16 | Unsupported | Full FP16 and full FSDP are fail-closed |
| LoRA | Supported | Supported | Conditional | Family target modules require exact model-data inspection and a real-model pilot |
| int8-LoRA | Supported | Supported | Unsupported | Every device must declare 8-bit support and provide compute capability 7.5 or newer |
| QLoRA | Supported | Supported | Unsupported | Every device must declare 4-bit support and provide compute capability 6.0 or newer |

“Supported” means the candidate can pass planner rules. It does not mean every
model, dataset, driver, or device combination has release evidence.

## Precision and quantization

| Path | Catalog behavior | Runtime evidence |
|---|---|---|
| BF16 | Selected when every participating device declares BF16 | Confirmed by preflight and pilot |
| FP16 full | Unsupported | No launch |
| FP16 adapter methods | Selectable when BF16 is not declared | Exact pilot required |
| int8 base load | int8-LoRA only, compute capability 7.5 or newer | Exact bitsandbytes pilot required |
| NF4 double quantization | QLoRA only, compute capability 6.0 or newer | Exact bitsandbytes pilot required |

## Model families

The target-module catalog currently names `llama`, `mistral`, `gemma`, and
`qwen`. Full fine-tuning does not need adapter target modules. Adapter methods
verify that every named module exists after loading the pinned model.

## Dataset and target

- File suffixes: JSONL, JSON, CSV, and text.
- Canonical schemas: text, prompt-completion, instruction-output, messages, and
  mixed input where normalization succeeds.
- Task: supervised fine-tuning only.
- Packing: unsupported.
- Enforced wall-time deadline: unsupported.
- Canonical compilation: every row, not only the profiling sample.

## Execution

- Backend: CUDA only.
- Managed concurrency: one Aptus job per local user and host across state roots.
- Runtime actions: dependency, model-data, preflight, pilot, train.
- Full-run resume: unsupported.
- Full-run output: unique no-clobber run-ID directory. Aptus does not make the
  directory immutable at the filesystem level.
- Export verification: structural safetensors file tree and provenance.
- Quality evaluation: not implemented as a target or gate.

## Future seams

ROCm, MPS, cloud providers, provider provisioning, evaluation policies, exporter
plugins, and MCP adapters are outside the current support contract.
