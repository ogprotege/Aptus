# Aptus Legacy Sandbox Verification

Date: 2026-07-21

The checks ran against a disposable copy of `HyperTune/` from Cursor's shell
sandbox. The audit runner itself does **not** enforce OS-level filesystem or
network isolation and now refuses to run without explicit acknowledgement.
Common credential environment variables were excluded. Registry-only checks
could receive host proxy variables, which may themselves carry credentials;
legacy-import probes did not receive those proxy variables. Dependency lifecycle
scripts were disabled. Raw commands, inherited environment-key names, timing,
output hashes, and bounded diagnostic previews are in `sandbox-results.jsonl`.

## Outcome

Nine checks ran. Two passed, six failed, and one was blocked.

Passed:

- `server.js` is valid JavaScript syntax under Node 26.5.0.
- `resource_scanner.py` completed its basic contract after a disposable probe
  supplied the missing `psutil` dependency. This supports ADAPT status, not
  direct reuse: the as-is check failed before execution because its environment
  could not satisfy the declared dependency set.

Failed:

- TypeScript 7.0.2 rejected the source tree at parse time. Several `.ts` files
  begin with Markdown headings, and `deploy/config_big.ts` contains embedded
  prose/configuration that is not TypeScript.
- npm could not produce a lockfile because
  `huggingface-api@^0.3.0` has no matching registry version. Independent
  metadata checks also returned E404 for `mcp-server@^0.1.0`.
- The pinned Python stack cannot resolve on Python 3.14:
  `transformers==4.35.0` requires `tokenizers<0.15,>=0.14`, for which no
  compatible Python 3.14 wheel exists.
- `script_generator_v2.py` first failed because Jinja2 is imported but absent
  from every relevant requirements file.
- A disposable Jinja2 shim exposed the next failure:
  `ScriptGenerator` calls `_get_transformers_lora_template`, which is not
  defined. The generator cannot construct even when the missing package is
  supplied.

Blocked:

- Test collection was not executed. Dependency resolution was dry-run only and
  failed, so no isolated project environment existed. Importing the suite would
  have measured host-package contamination rather than the legacy project.
  Static parsing independently confirms the `core_optimizer.py:1021`
  `IndentationError`.

## Static corroboration

The read-only reference analyzer found:

- 143 script files: 71 Python and 72 JavaScript/TypeScript.
- Three Python parse failures:
  `Complete Guide to Building & Deploying.py` (Markdown under a `.py`
  extension), `src/python/core_optimizer.py`, and
  `src/python/script_generator.py`.
- 40 missing relative imports across the script tree.

## Checks intentionally not attempted

- No service was started. Dependency, parse, and construction gates failed
  before a service smoke test could be meaningful.
- No generated training bundle was executed. Both Python generators failed
  before valid artifact production.
- No model weights or tokenizers were downloaded.
- No GPU test or paid cloud job ran.
- No remote model code was trusted or executed.

These are blocked checks, not silent passes. A future Aptus implementation must
earn them from a clean dependency graph and validated artifact generator.
