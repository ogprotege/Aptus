# Aptus legacy static TypeScript/JavaScript forensic report

> **Documentation status:** Archived evidence
>
> **Applies to:** Dated forensic review of the removed legacy `HyperTune/` tree
>
> **Last reviewed:** 2026-08-06
>
> **Next scheduled review:** 2027-08-06, or when provenance or reproduction paths change
>
> **Historical warning:** The body below is preserved point-in-time evidence,
> not current web implementation guidance. Start with the [audit index](README.md)
> or [current capabilities](../../product/current-capabilities.md).

## Scope and confidence

This report statically inspects the legacy `HyperTune/` source folder and the generated inventory, duplicate, version-family, and reference-map evidence. This static track did not install a package or execute legacy commands. Later disposable checks are reported separately in `sandbox-summary.md`; they confirm that `server.js` parses but the TypeScript project and dependency lock do not.

Inventory-derived scope: 70 `.ts` files and 2 `.js` files. Nineteen TypeScript files are empty; the generated empty-content cluster contains 23 files across all formats (`docs/audits/aptus-legacy/duplicate-clusters.json:303-330`). The reference generator records JavaScript/TypeScript parse status as `not_checked`, so this report's syntax and contract findings come from direct source inspection (`docs/audits/aptus-legacy/reference-map.json:1371-1419,2541-2567`). The subsequent TypeScript 7.0.2 sandbox check independently failed on invalid source text.

Confidence labels:

- **High**: directly observed path, content, import, export, or package-script fact.
- **Medium**: runtime consequence inferred from normal Node/TypeScript behavior without execution.
- **Low**: product-value or algorithm-quality judgment needing validation.

## Executive finding

**Observed — High confidence.** The tree expresses several intended topologies but does not contain one coherent TypeScript/JavaScript application:

- The package-selected topology is an Express HTTP façade that dispatches custom function calls to Python workers: `main` and both package scripts select `server.js` (`HyperTune/package.json:5-8`).
- Four separate files independently describe stdio MCP servers: `HyperTune/src/index.ts:6-69`, `HyperTune/src/index_v2.ts:8-49`, `HyperTune/src/mcp_index.ts:10-197`, and `HyperTune/src/mem_index.ts:18-111`.
- A serverless/Next-style API is suggested by `HyperTune/api/optimize.ts:1-61`; an Express comparison API is suggested by named handlers in `HyperTune/api/comparison.ts:27-244`.
- A provider layer is suggested by `HyperTune/src/integrations/registry.ts:38-115`, while a separate, incompatible provider draft exists under `HyperTune/src/providers/`.

Only `server.js` is selected by package metadata. No package script builds or starts TypeScript, `tsconfig.json` is not JSON, and the package omits the compiler and the external packages imported by the TypeScript graph (`HyperTune/package.json:6-22`; `HyperTune/tsconfig.json:1-2`).

**Observed — High confidence.** Even the selected HTTP path is not operational as checked in:

- All seven Python invocations use `scriptPath: './python'` (`HyperTune/server.js:74-82,99-107,130-144,179-193,217-232,274-294,395-408`).
- The worker files are under `HyperTune/src/python/`, not `HyperTune/python/` (for example, `HyperTune/src/python/register_model.py`, `register_dataset.py`, `train.py`, and `export_model.py`; inventory entries at `docs/audits/aptus-legacy/inventory.jsonl:187-207`).
- Therefore, when `npm start` runs from the package directory, the HTTP process may reach `listen` if its declared dependencies are present, but every worker-backed operation resolves its script from an absent directory. **Runtime consequence — Medium confidence**, because execution was intentionally not attempted.

**Hypothesis — Medium confidence.** The source is an accumulation of successive prototypes—HTTP/Python bridge, stdio MCP, serverless API, and provider marketplace—rather than forks derived from a maintained canonical core.

## Entrypoints and observable API drift

### Package-selected HTTP entrypoint

**Observed — High confidence.**

- `server.js` accepts only `POST /mcp` (`HyperTune/server.js:483-487`) and dispatches a body shaped as `{ function_name, parameters }` (`HyperTune/server.js:18-23`).
- Its function vocabulary is `register_model`, `register_dataset`, `optimize_layers`, `decompose_weights`, `setup_reft`, `start_training`, `get_training_status`, and `export_model` (`HyperTune/server.js:23-46`).
- It is a custom request/response dispatcher, not the stdio transport used by the TypeScript MCP drafts.
- CORS is unrestricted by configuration, and the route has no authentication or rate limiting (`HyperTune/server.js:10-13,483-487`).
- Job state is process-local memory (`HyperTune/server.js:15-16,258-270`). A restart loses all jobs.
- The stored job has `config: parameters` but no top-level `method` (`HyperTune/server.js:261-268`); status rendering later tests `job.method !== 'reft'` (`HyperTune/server.js:336-352`), so that condition does not read the submitted method.

The checked-in JavaScript request script targets a nonexistent `/sse` route and sends a third request shape, `{ request: { functions: [...] } }`, with an `optimize` function (`HyperTune/scripts/test-request.js:4-22`). It cannot exercise the package-selected API.

### Disconnected TypeScript entrypoints

**Observed — High confidence.**

- `src/index.ts` offers `optimize`, `generate_script`, and `generate_config` over stdio (`HyperTune/src/index.ts:6-64`).
- `src/index_v2.ts` offers generated method tools plus model-specific tools (`HyperTune/src/index_v2.ts:9-46`).
- `src/mcp_index.ts` offers `analyze_model`, `run_fine_tuning`, and `evaluate_fine_tuned_model` (`HyperTune/src/mcp_index.ts:17-157`).
- `src/mem_index.ts` offers `start_fine_tuning` and `check_job_status`; its `analyze_model` entry is only an object containing a comment, with no schema or handler (`HyperTune/src/mem_index.ts:24-27`).
- None is referenced by `package.json`; there is no build script, TypeScript runtime, or emitted JavaScript path.
- `api/optimize.ts` is a default Next handler, but `next` is absent from dependencies (`HyperTune/api/optimize.ts:2-7`; `HyperTune/package.json:10-22`).
- `api/comparison.ts` exports named Express handlers rather than a registered Express route or default serverless handler (`HyperTune/api/comparison.ts:30,144,205`).

## Build blockers, missing imports, and invalid configuration

### Invalid source/configuration

**Observed — High confidence.**

- `tsconfig.json` contains two hash-prefixed text lines, not a JSON object (`HyperTune/tsconfig.json:1-2`).
- Eighteen `.ts` files contain shell/Python-style `#` lines that are invalid TypeScript. Seventeen begin that way, including:
  - `HyperTune/src/auth/api_key.ts:1`
  - `HyperTune/src/auth/rate_limiting.ts:1`
  - `HyperTune/src/formulas/{index,alpha,target_modules,training_steps,warmup}.ts:1`
  - `HyperTune/src/hypertuner/methodSelector.ts:1`
  - `HyperTune/src/hypertuner/methods/base.ts:1-2`
  - `HyperTune/src/mcp_index.ts:1-2`
  - `HyperTune/src/mem_index.ts:1-2`
  - `HyperTune/src/integrations/{registry,training/huggingface,training/runpod,deployment/vercel,framework/langchain}.ts:1-2`
- `deploy/config_big.ts` starts as TypeScript, then contains bare Markdown fences, prose, Dockerfile text, Python, and requirements content (`HyperTune/deploy/config_big.ts:1-8,54-110,270-314`).
- `package.json` has no `build` or `test` script, although the embedded Dockerfile draft calls `npm run build` (`HyperTune/package.json:6-9`; `HyperTune/deploy/config_big.ts:73-87`).
- There is no lockfile in the legacy folder, so dependency resolution is not reproducible from the checked-in evidence.
- The Prisma schema is empty (`docs/audits/aptus-legacy/inventory.jsonl:104`), while authentication code constructs `PrismaClient` and queries `apiKey` relations (`HyperTune/src/auth/api_key.ts:3-18`).

### Absent local modules

The generated reference map contains 40 missing import edges overall; one is Python, leaving 39 recorded JavaScript/TypeScript edges. Direct reading adds a multiline import the extractor missed—`../utils/visualization` in `HyperTune/src/integrations/reft_integration.ts:6-11`—so the TypeScript/JavaScript graph has **at least 40 absent local imports**.

High-impact groups:

- `api/comparison.ts`: three wrong relative roots (`../model-database`, `../methods`, `../hypertuner/task-configs`) (`HyperTune/api/comparison.ts:3-5`; generated evidence at `docs/audits/aptus-legacy/reference-map.json:1394-1414`).
- `methods.ts`: all seven implementation modules are absent (`HyperTune/src/methods.ts:4-10`; `docs/audits/aptus-legacy/reference-map.json:3078-3126`).
- `optimizer.ts`: `./utils/estimation` and `./output/huggingface` are absent (`HyperTune/src/optimizer.ts:4,11`; `docs/audits/aptus-legacy/reference-map.json:3306-3312,3355-3361`).
- `mcp_index.ts`: LoRA, DoRA, and QLoRA class modules are absent (`HyperTune/src/mcp_index.ts:6-8`; `docs/audits/aptus-legacy/reference-map.json:2992-3012`).
- `mem_index.ts`: `./providers/runpod` is absent (`HyperTune/src/mem_index.ts:6-7`; `docs/audits/aptus-legacy/reference-map.json:3042-3055`).
- `hypertuner/methods/base.ts`: `../utils/shell` is absent; the same file imports its own interface from itself (`HyperTune/src/hypertuner/methods/base.ts:4-15`; `docs/audits/aptus-legacy/reference-map.json:2381-2396`).
- `models/index.ts`: `./premium/mixtral` is absent; statically imported `llama.ts` and `gemma.ts` exist but are empty (`HyperTune/src/models/index.ts:58-76`; inventory at `docs/audits/aptus-legacy/inventory.jsonl:169-176`).
- `integrations/reft_integration.ts`: `../methods/reft_enhanced` and `../utils/visualization` are absent (`HyperTune/src/integrations/reft_integration.ts:2-11`; the first is recorded at `docs/audits/aptus-legacy/reference-map.json:2679-2684`).
- `integrations/registry.ts`: 20 of its 24 concrete provider imports are absent—four training, five deployment, seven model, and four framework adapters (`HyperTune/src/integrations/registry.ts:6-36`; `docs/audits/aptus-legacy/reference-map.json:2729-2882`).

### Package/import mismatch

**Observed — High confidence.** TypeScript/JavaScript imports external packages absent from `package.json`: `@modelcontextprotocol/sdk`, `next`, `@prisma/client`, `rate-limiter-flexible`, `axios`, and `node-fetch` (`HyperTune/src/index.ts:2`; `HyperTune/api/optimize.ts:2`; `HyperTune/src/auth/rate_limiting.ts:3-5`; `HyperTune/src/integrations/training/huggingface.ts:3-4`; `HyperTune/scripts/test-request.js:2`). TypeScript itself and a TypeScript runner are also absent. Conversely, declared `mcp-server` and `huggingface-api` are not imported by the inspected runtime (`HyperTune/package.json:10-18`).

## API and internal contract drift

**Observed — High confidence.**

- `methods.ts` exports only `createTuningMethods`; it does not export `methodRegistry` or `PeftMethod` (`HyperTune/src/methods.ts:12-170`). `optimizer.ts`, every model-aware formula, and both output modules import those nonexistent exports (`HyperTune/src/optimizer.ts:2`; `HyperTune/src/formulas/learning_rate.ts:2-3`; `HyperTune/src/output/command_line.ts:2`).
- `models/index.ts` expects `llamaModels` and `gemmaModels` from empty files, then immediately iterates them (`HyperTune/src/models/index.ts:59-66`).
- `formulas/index.ts` re-exports `calculateOptimalLearningRate`, `calculateOptimalBatchSize`, and `calculateOptimalWeightDecay`, but the target files export `calculateLearningRate`, `calculateBatchSize`, and `calculateWeightDecay` (`HyperTune/src/formulas/index.ts:4-7`; `HyperTune/src/formulas/learning_rate.ts:6`; `HyperTune/src/formulas/batch_size.ts:6`; `HyperTune/src/formulas/weight_decay.ts:6`). It also calls those names locally without importing bindings and references undefined `MethodType` (`HyperTune/src/formulas/index.ts:16-67`).
- `api/optimize.ts` imports `rateLimit`, but the module exports `rateLimiter` (`HyperTune/api/optimize.ts:5,13-16`; `HyperTune/src/auth/rate_limiting.ts:25`). It passes a string to `validateApiKey` and expects `{ valid, tier }`, while the implementation is three-argument middleware over `(req, res, next)` and returns HTTP responses (`HyperTune/api/optimize.ts:29-36`; `HyperTune/src/auth/api_key.ts:8-35`). `src/index.ts` makes the same incompatible string call and treats the result as boolean (`HyperTune/src/index.ts:19-25`).
- `index_v2.ts` creates method tools before separately registering LLM tools (`HyperTune/src/index_v2.ts:30-42`). The `fine_tune` handler closes over its private `tools` object and dereferences `tools.detect_llm`, `tools.check_method_compatibility`, and `tools.generate_optimal_config`, none of which `createTuningMethods` adds (`HyperTune/src/methods.ts:12-44,71-100`).
- `integrations/types.ts` references undeclared types such as `TrainingJobConfig`, `JobStatus`, `DeploymentConfig`, and `Model` (`HyperTune/src/integrations/types.ts:17-39`). The concrete adapters therefore do not have a closed type contract even before their missing dependencies and invalid `#` lines are addressed.
- Two incompatible model schemas coexist: `Model` uses `parameters`, `architecture`, and `layers` (`HyperTune/src/models/index.ts:2-13`); `ModelInfo` uses `size`, `family`, and `numLayers` (`HyperTune/src/model-database.ts:2-22`). `api/comparison.ts` is written for `ModelInfo` plus a nonexistent `methodsDatabase`, while `optimizer.ts` is written for `Model` plus a nonexistent `methodRegistry`.
- `task-configs.ts` takes a shared configuration object, mutates its batch size, rank, and target layers, then shallow-copies it (`HyperTune/src/hypertuner/task-configs.ts:368-405`). Repeated calls can compound prior model-specific adjustments. Its four comparison metrics are fixed constants regardless of arguments (`HyperTune/src/hypertuner/task-configs.ts:442-463`).
- The active learning-rate formula injects nondeterminism through `Math.random()` (`HyperTune/src/formulas/learning_rate.ts:60-64`), so identical optimization inputs need not produce identical output.

## Unsafe command and code generation

**Observed generation risk — High confidence; exploitability — Medium confidence because no generated string was executed.**

- `hypertuner/methods/base.ts` feeds interpolated model paths, dataset paths, output paths, numeric/config fields, and joined target modules into shell command strings without quoting or argument separation. It also shells out for `mkdir` and `cp` (`HyperTune/src/hypertuner/methods/base.ts:21-28,45-74`). The executor import is currently absent, but restoring it without redesign would expose command injection and option injection.
- `output/command_line.ts` interpolates model/method names and every config entry into runnable commands without shell escaping (`HyperTune/src/output/command_line.ts:30-34,97-114`). JSON-stringifying target modules and replacing quotes is not shell-safe (`HyperTune/src/output/command_line.ts:127-134`).
- `createReftPythonCommand` escapes only double quotes inside JSON; model, dataset, and output paths are unquoted and shell metacharacters remain active (`HyperTune/src/integrations/reft_integration.ts:282-296`).
- `VercelProvider` interpolates `config.task` and `config.modelPath` directly into generated JavaScript string literals (`HyperTune/src/integrations/deployment/vercel.ts:108-143`). This is generated-code injection, not shell injection, but has the same trust-boundary problem if user-controlled values are deployed.

By contrast, the package-selected `server.js` supplies Python arguments as an array to `python-shell` (`HyperTune/server.js:74-82,274-294`). That avoids the direct shell-string pattern above, although arbitrary filesystem/model inputs still cross into Python without validation.

## Duplicate and fork families

**Observed — High confidence.**

- Generated evidence identifies 38 exact-content clusters and 30 normalized version families overall (`docs/audits/aptus-legacy/duplicate-clusters.json:1-3`; `docs/audits/aptus-legacy/version-families.json:1-3,246-248`).
- TypeScript-relevant version families are:
  - `src/index.ts` / `src/index_v2.ts` (`docs/audits/aptus-legacy/version-families.json:142-147`).
  - `batch_size.ts` / `batch_size_v2.ts`, `rank.ts` / `rank_v2.ts`, `weight_decay.ts` / `weight_decay_v2.ts`, and `learning_rate.ts` / `learning_rate_2.ts` (`docs/audits/aptus-legacy/version-families.json:110-139`).
- The optimizer imports the unversioned formula files (`HyperTune/src/optimizer.ts:5-8`); no code imports the `_2`/`_v2` variants. The alternatives differ materially—for example, QLoRA multiplies the active learning rate by `1.5`, while the alternate multiplies its different base by `12` (`HyperTune/src/formulas/learning_rate.ts:26-34`; `HyperTune/src/formulas/learning_rate_2.ts:15-27`).
- `src/formulas/learning_parameters.ts` concatenates six formula modules, repeatedly redeclaring the same imports (`HyperTune/src/formulas/learning_parameters.ts:1-3,104-106,186-188,258-260,374-376,453-455`). It is byte-identical to `Random/hyper-p-long.md` (`docs/audits/aptus-legacy/duplicate-clusters.json:132-138`). Identity is proven; copy direction is not. The concatenated module boundaries support the hypothesis that the `.ts` file is a document dump rather than a canonical module.
- `src/types.ts` is byte-identical to `src/hparam_methods_reference.md` (`docs/audits/aptus-legacy/duplicate-clusters.json:123-129`).
- `server.js` is byte-identical to `server 2.txt` (`docs/audits/aptus-legacy/duplicate-clusters.json:195-201`).
- The 23-file zero-byte cluster mixes route placeholders, scripts, models, premium methods/models, docs, Prisma, and UI content (`docs/audits/aptus-legacy/duplicate-clusters.json:303-330`).

## Proven empty files and nonempty stubs

### Empty TypeScript files

**Observed — High confidence.** Nineteen `.ts` files are zero bytes:

- API/auth/subscription: `api/auth/{create-key,verify-key}.ts`; `api/subscription/{check,create,update}.ts`.
- Scripts/auth barrel: `scripts/{seed-models,test-optimizations}.ts`; `src/auth/index.ts`.
- Premium methods: `src/methods/premium/{alora,hydrora,moe_lora,priolora}.ts`.
- Models: `src/models/{llama,gemma}.ts`; `src/models/premium/{claude,gemini,gpt,grok}.ts`.
- Misspelled output placeholder: `src/output/huggeface.ts`.

The inventory proves the zero-byte state (`docs/audits/aptus-legacy/inventory.jsonl:70-76,114-116,122,163-176,181`).

### Nonempty stubs

**Observed — High confidence.**

- Model-size detection falls back to `7` billion parameters; dataset read failure falls back to `1000` examples (`HyperTune/src/mcp_index.ts:162-188`).
- LLM structure inference always returns `null` (`HyperTune/src/llm_integrations.ts:205-210`).
- Dynamic config loading always returns `null` (`HyperTune/src/model-database-update.ts:476-491`).
- ReFT availability always returns `true` without checking the environment (`HyperTune/src/integrations/reft_integration.ts:299-309`).
- Method comparison metrics are hardcoded constants (`HyperTune/src/hypertuner/task-configs.ts:442-463`).
- The embedded Python in `deploy/config_big.ts` explicitly describes interface-only implementations and returns simulated success data (`HyperTune/deploy/config_big.ts:109-140`); the file is not valid TypeScript in any case.

## Disposition hypotheses

These are migration hypotheses, not observed facts.

- **ARCHIVE — Medium confidence:** `HyperTune/src/types.ts` and `HyperTune/deploy/config.ts` preserve minimal contract/deployment vocabulary but are too incomplete and drifted for direct reuse (`HyperTune/src/types.ts:1-9`; `HyperTune/deploy/config.ts:1-53`). The decision ledger therefore records both as ARCHIVE, not KEEP.
- **ADAPT — Medium confidence:** `HyperTune/src/model-database.ts`, `HyperTune/src/model-database-update.ts`, and `HyperTune/src/hypertuner/task-configs.ts`. They contain substantial structured data and helper logic, but need provenance checks, one canonical model schema, immutable configuration handling, and replacement of fixed metrics/placeholders.
- **ADAPT — Medium confidence:** the pure formula modules under `HyperTune/src/formulas/`, after selecting one version per family, removing nondeterminism, restoring one method contract, and adding calibration tests. Their numerical claims are **Low-confidence** until tied to sources or benchmarks.
- **ADAPT — Medium confidence:** `HyperTune/src/integrations/types.ts` as a starting vocabulary, after defining every referenced type and choosing one provider method naming scheme. Treat concrete adapters as examples until their remote APIs are independently verified.
- **ADAPT or ARCHIVE — Medium confidence:** `HyperTune/server.js`. Preserve its route/job vocabulary only if Aptus intentionally retains an HTTP-to-Python bridge; otherwise archive it after extracting requirements. It is not a substitute for a standards-compliant MCP transport.
- **ARCHIVE — High confidence:** the four disconnected TypeScript entrypoint drafts, speculative provider registry/adapters, and exact text copies such as `server 2.txt`. They are useful provenance but should not remain competing production roots.
- **ARCHIVE or DISCARD — High confidence:** preserve `deploy/config_big.ts` and `src/formulas/learning_parameters.ts` only as historical evidence; discard all 19 empty TypeScript placeholders and the misspelled empty `src/output/huggeface.ts`. For the ADAPT-labeled command-generation seam, retain only the output contract and reject the current unsafe interpolation; any execution must use validated structured plans and argument arrays without a shell.

## Overall confidence

- **High:** package-selected entrypoint, invalid configuration/source text, missing/empty files, import/export mismatches, API-shape divergence, duplicate families, and unsafe interpolation are directly evidenced.
- **Medium:** the selected HTTP process could listen after dependency installation but cannot locate its workers when started from the package directory; this follows from paths but was not executed.
- **Low:** formula quality, model metadata accuracy, provider endpoint validity, and final KEEP/ADAPT choices require current product requirements, source provenance, and benchmark or integration tests.
