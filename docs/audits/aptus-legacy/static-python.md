# Aptus legacy Python static forensic audit

> **Documentation status:** Archived evidence
>
> **Applies to:** Dated forensic review of the removed legacy `HyperTune/` tree
>
> **Last reviewed:** 2026-08-06
>
> **Next scheduled review:** 2027-08-06, or when provenance or reproduction paths change
>
> **Historical warning:** The body below is preserved point-in-time evidence,
> not current Python implementation guidance. Start with the
> [audit index](README.md) or [current capabilities](../../product/current-capabilities.md).

## Scope and proof standard

This report covers the Python code and tests under `HyperTune/` and the generated evidence under `docs/audits/aptus-legacy/`. It is the read-only static assessment track. That track did not import legacy modules or install packages. Later disposable checks, recorded in `sandbox-summary.md`, corroborate the optimizer/test/generator failures and run narrowly bounded salvage probes; no generated training script or model training ran.

The generated reference map contains 71 Python records: 68 are marked as parse-passed and three as parse-failed. “Parse-passed” means only that the audit parser accepted the file. It does **not** prove imports, object construction, dependency compatibility, generated-output validity, training, evaluation, or an end-to-end request. The distinction matters here because several parse-passed files have deterministic import, attribute, schema, or computation-graph failures on inspection (`docs/audits/aptus-legacy/reference-map.json:1366-1369`, `docs/audits/aptus-legacy/reference-map.json:4023-4026`, `docs/audits/aptus-legacy/reference-map.json:4554-4557`).

No Python runtime path is runtime-proven by this audit. No prior test-result, training-result, benchmark, generated-script compilation result, or model artifact appears in the supplied audit evidence. Confidence labels below mean:

- **High:** directly established by syntax evidence, exact duplication hashes, or an unambiguous local code contract.
- **Medium:** strongly implied by local code but dependent on third-party version or model behavior not executed here.
- **Low:** a preservation hypothesis requiring provenance or runtime investigation.

## Executive finding

Aptus should not treat the legacy Python as a working optimization or training backend. There is no intact path from either REST API or the workflow class through optimization and script generation:

- The central optimizer does not parse (`docs/audits/aptus-legacy/reference-map.json:3633-3642`).
- The v1 script generator does not parse (`docs/audits/aptus-legacy/reference-map.json:4512-4521`).
- The parse-valid v2 generator cannot be constructed because its initializer calls template-builder methods that do not exist (`HyperTune/src/python/script_generator_v2.py:11-41`, `HyperTune/src/python/script_generator_v2.py:315-347`).
- The optimizer’s Optuna studies do not train or evaluate a model. They maximize hand-written formulas, then label the formula value `expected_performance` (`HyperTune/src/python/core_optimizer.py:97-149`, `HyperTune/src/python/core_optimizer.py:203-208`).
- Several advertised tuning/evaluation paths return fixed success metrics or fixed scores without doing the advertised work (`HyperTune/src/python/tune_service.py:22-45`, `HyperTune/src/python/tune_service.py:74-100`, `HyperTune/src/python/reft_enhanced.py:348-412`).

The defensible recovery strategy is to keep a small set of data-inspection and real trainer ideas, rebuild one canonical Aptus package and API contract, and archive or discard the duplicated orchestration and heuristic “optimizer” surface.

## 1. Parse failures and their blast radius

### 1.1 Mislabeled prose file — High confidence

`HyperTune/Complete Guide to Building & Deploying.py` is Markdown prose beginning on line 3, not Python (`HyperTune/Complete Guide to Building & Deploying.py:1-10`). The parser reports an unterminated string at line 3 (`docs/audits/aptus-legacy/reference-map.json:3-12`). It is byte-identical to two Markdown files, so the `.py` copy has no independent code value (`docs/audits/aptus-legacy/duplicate-clusters.json:77-84`).

### 1.2 Central optimizer is unimportable — High confidence

`HyperTune/src/python/core_optimizer.py` has an unexpected indent at line 1021 (`docs/audits/aptus-legacy/reference-map.json:3633-3642`). The source places a second f-string at an indentation that is not part of a parenthesized expression (`HyperTune/src/python/core_optimizer.py:1017-1024`).

This blocks every direct consumer before any dependency or behavior can be tested:

- The root REST app imports it at module load (`HyperTune/hyperparameter_mcp.py:15-29`).
- The workflow/CLI class imports it at module load (`HyperTune/src/python/hyperparameter_mcp.py:10-20`).
- The alternate FastAPI app imports it at module load (`HyperTune/api/FastAPI/main.py:14-24`).
- Optimizer and integration tests import it during collection (`HyperTune/tests/test_optimizer.py:7-17`, `HyperTune/tests/integration/test_workflow.py:8-17`).

### 1.3 v1 generator is unimportable — High confidence

The v1 generator opens a large outer f-string at line 222 and embeds a literal empty dictionary at line 308; `{}` is parsed as an empty f-string expression, producing the recorded syntax error (`HyperTune/src/python/script_generator.py:222-228`, `HyperTune/src/python/script_generator.py:301-313`, `docs/audits/aptus-legacy/reference-map.json:4512-4521`). The alternate FastAPI app imports this broken generator, so fixing only the optimizer would still not make that app importable (`HyperTune/api/FastAPI/main.py:14-24`).

## 2. Orchestrator and API drift

There are three incompatible Python orchestration surfaces, none of which implements an MCP transport. Two are FastAPI REST applications and one is an in-process workflow/CLI class (`HyperTune/hyperparameter_mcp.py:20-29`, `HyperTune/api/FastAPI/main.py:19-24`, `HyperTune/src/python/hyperparameter_mcp.py:20-40`).

### 2.1 Root REST app — High confidence

- Its request accepts precomputed dataset statistics and the field `model_type` (`HyperTune/hyperparameter_mcp.py:43-56`), while the alternate app uses `model_id` and a different schema (`HyperTune/api/FastAPI/main.py:46-65`), and the workflow class accepts a dataset path (`HyperTune/src/python/hyperparameter_mcp.py:99-105`).
- It calls `script_generator.generate(...)` (`HyperTune/hyperparameter_mcp.py:116-123`), but the v2 class exposes `generate_script(...)` (`HyperTune/src/python/script_generator_v2.py:44-53`).
- The CLI offers script format `shell` (`HyperTune/hypertune_cli.py:29-31`), whereas the v2 generator registers `bash`, not `shell` (`HyperTune/src/python/script_generator_v2.py:36-42`).
- The model registry defaults to `../../../Scrapers/data/base_models.json` and silently becomes empty when that file is absent (`HyperTune/src/python/model_registry.py:16-34`). Static inventory contains no `base_models.json`; the only nearby manual data is at a different path (`docs/audits/aptus-legacy/inventory.jsonl:23`).

### 2.2 Workflow/CLI class — High confidence

- It imports and constructs `ResourceScanner`, but the resource module defines `ResourceInfo`, not `ResourceScanner` (`HyperTune/src/python/hyperparameter_mcp.py:14-18`, `HyperTune/src/python/hyperparameter_mcp.py:48-60`, `HyperTune/src/python/resource_scanner.py:15-30`).
- `DatasetAnalyzer.analyze()` returns a `DatasetStats` Pydantic object (`HyperTune/src/python/dataset_analyzer.py:41-55`, `HyperTune/src/python/dataset_analyzer.py:141-149`), but the workflow subscripts it as a dictionary using nonexistent keys `num_examples` and `avg_sequence_length` (`HyperTune/src/python/hyperparameter_mcp.py:131-148`). The actual fields are `total_examples` and `avg_tokens_per_example` (`HyperTune/src/python/dataset_analyzer.py:13-22`).
- It expects resource keys `available_gpu_memory_gb` and `gpu_count` (`HyperTune/src/python/hyperparameter_mcp.py:145-146`); `ResourceInfo.to_dict()` returns `usable_gpu_memory_gb` and omits `gpu_count` (`HyperTune/src/python/resource_scanner.py:239-248`).
- Its “full workflow” analyzes the same dataset twice—once directly and once inside `optimize_hyperparameters()`—which can yield different random samples because dataset sampling is unseeded (`HyperTune/src/python/hyperparameter_mcp.py:219-229`, `HyperTune/src/python/dataset_analyzer.py:67-85`).

### 2.3 Alternate FastAPI app — High confidence

- It imports both parse-failed central files, so the app cannot start as written (`HyperTune/api/FastAPI/main.py:14-24`).
- Model-load failure returns an `HTTPException` object instead of raising it (`HyperTune/api/FastAPI/main.py:71-79`). The end-to-end route then assumes the result has `.parameters` (`HyperTune/api/FastAPI/main.py:249-265`).
- Parameter units are mixed: `num_parameters` may be a raw count, while the fallback calculation explicitly converts to billions; both flow into `model_size_billions` unchanged (`HyperTune/api/FastAPI/main.py:95-107`, `HyperTune/api/FastAPI/main.py:262-267`).
- `model_size_billions` is optional in the API request but required by `OptimizationContext` (`HyperTune/api/FastAPI/main.py:46-56`, `HyperTune/src/python/core_optimizer.py:9-20`).
- Manual `gpu_count` is assigned dynamically to `ResourceInfo`, but `to_dict()` does not serialize it, so the override is lost (`HyperTune/api/FastAPI/main.py:156-170`, `HyperTune/src/python/resource_scanner.py:239-248`).

## 3. Optimizer correctness: search over formulas, not empirical optimization

### 3.1 No trial observes a model, dataset, loss, or metric — High confidence

Each Optuna objective samples values and returns a deterministic arithmetic score. No objective loads a model, consumes examples, runs training, evaluates a validation set, or accepts a caller-provided metric:

- LoRA: rank × epochs × batch size (`HyperTune/src/python/core_optimizer.py:100-145`).
- QLoRA: the same pattern multiplied by fixed NF4/double-quantization factors (`HyperTune/src/python/core_optimizer.py:214-271`).
- Full fine-tuning: effective batch × epochs × inverse learning rate (`HyperTune/src/python/core_optimizer.py:353-397`).
- Other PEFT methods: fixed method/module priors × epochs × batch size (`HyperTune/src/python/core_optimizer.py:478-599`).
- DPO: rank × epochs × a hand-coded beta bucket × a reference-model factor (`HyperTune/src/python/core_optimizer.py:776-828`).

Optuna is therefore only a randomized enumerator for closed-form heuristics. Calling the result “optimized” or “expected performance” is unsupported. For LoRA and QLoRA, sampled learning rate, dropout, and alpha do not affect the performance objective at all (`HyperTune/src/python/core_optimizer.py:100-107`, `HyperTune/src/python/core_optimizer.py:130-145`, `HyperTune/src/python/core_optimizer.py:214-225`, `HyperTune/src/python/core_optimizer.py:249-271`). Tied objective values can select those fields arbitrarily, and no sampler seed is set (`HyperTune/src/python/core_optimizer.py:147-152`, `HyperTune/src/python/core_optimizer.py:273-278`).

`expected_performance` is simply `study.best_value` (`HyperTune/src/python/core_optimizer.py:203-208`, `HyperTune/src/python/core_optimizer.py:342-347`). It is neither calibrated nor bounded to a metric range; the LoRA formula can exceed 1 by a large factor. Explanatory prose consequently overstates evidence by saying parameters “are optimized” (`HyperTune/src/python/core_optimizer.py:951-964`).

### 3.2 Feasibility handling is internally inconsistent — High confidence

An over-memory trial returns negative infinity, but there is no “no feasible trials” guard before reading `study.best_params` (`HyperTune/src/python/core_optimizer.py:118-149`, `HyperTune/src/python/core_optimizer.py:151-162`). The memory formula used after selection differs from the objective formula: the objective includes base + three adapter-memory units + tiny batch memory, while the reported result uses base + two adapter-memory units + a fixed 1 GB overhead (`HyperTune/src/python/core_optimizer.py:118-128`, `HyperTune/src/python/core_optimizer.py:159-162`). A configuration can therefore be accepted under one estimate and reported under another.

The same pattern recurs for QLoRA and DPO (`HyperTune/src/python/core_optimizer.py:236-247`, `HyperTune/src/python/core_optimizer.py:285-288`, `HyperTune/src/python/core_optimizer.py:793-804`, `HyperTune/src/python/core_optimizer.py:847-851`).

## 4. Memory, precision, and quantization logic

### 4.1 Memory estimates are dimensionally incomplete — High confidence

- Base-model memory is fixed at 2 GB per billion parameters (`HyperTune/src/python/core_optimizer.py:911-915`).
- LoRA memory is inferred only from `rank / 4096`, independent of layer count, target modules, hidden dimensions, dtype, gradients, or optimizer-state dtype (`HyperTune/src/python/core_optimizer.py:917-922`).
- Batch memory counts four bytes per token but omits hidden states, layers, attention matrices, backward activations, and temporary kernels (`HyperTune/src/python/core_optimizer.py:118-124`).
- Full fine-tuning gives Adam optimizer state only one base-model-memory unit despite the comment saying two moment tensors, and its activation formula uses “billions” as a small scalar rather than parameter or architecture dimensions (`HyperTune/src/python/core_optimizer.py:370-380`, `HyperTune/src/python/core_optimizer.py:423-428`).

These estimates are sizing heuristics, not safe capacity checks.

### 4.2 QLoRA base memory is undercounted — High confidence

The code starts from an fp16 estimate of 2 GB/B, then divides by eight for 4-bit weights (`HyperTune/src/python/core_optimizer.py:236-243`). Moving from 16-bit to 4-bit is a fourfold weight-storage reduction, not eightfold; the formula implies roughly two bits per parameter before even considering quantization metadata, dequantization buffers, adapters, activations, and CUDA overhead. The same `/ 8` appears in the reported result (`HyperTune/src/python/core_optimizer.py:285-288`).

The optimizer selects bitsandbytes-style 4-bit settings without checking CUDA/bitsandbytes support (`HyperTune/src/python/core_optimizer.py:305-328`). The dependency is pinned without platform markers (`HyperTune/requirements_v2.txt:7-12`), so CPU, Metal, and unsupported CUDA environments are not screened.

### 4.3 GPU capacity and precision are conflated — High confidence

`ResourceInfo` sums all GPU VRAM into one pool and applies an 85% factor (`HyperTune/src/python/resource_scanner.py:174-191`). The optimizer then treats that aggregate as a single feasibility limit, without checking sharding strategy or per-device batch placement.

The LoRA result hardcodes bf16 by model family even when bf16 CUDA is unavailable (`HyperTune/src/python/core_optimizer.py:42-62`, `HyperTune/src/python/core_optimizer.py:173-192`). Other branches check CUDA bf16 support, creating inconsistent precision policy (`HyperTune/src/python/core_optimizer.py:305-309`, `HyperTune/src/python/core_optimizer.py:439-454`). The standalone LoRA trainer also always requests bfloat16 and `device_map="auto"` (`HyperTune/src/hypertuner/training/lora_trainer.py:26-34`), while the generic trainer always loads fp16 before deciding whether the device is CPU (`HyperTune/src/python/train.py:52-60`, `HyperTune/src/python/train.py:111-113`).

### 4.4 Resource recommendations contain concrete bugs — High confidence

- The “round down to power of two” expression computes `2**bit_length - 1`, yielding values such as 7, 15, or 31 rather than powers of two (`HyperTune/src/python/resource_scanner.py:227-237`).
- The macOS parser recognizes only `MB` (`HyperTune/src/python/resource_scanner.py:153-167`), while its own test supplies `"16 GB"` and expects 16 × 1024 MB (`HyperTune/tests/test_resource_scanner.py:138-167`).
- Total, not currently free, VRAM is used (`HyperTune/src/python/resource_scanner.py:38-44`, `HyperTune/src/python/resource_scanner.py:174-191`).

## 5. Script-generator validity

### 5.1 Neither generator is usable as written — High confidence

The v1 generator is syntactically invalid as described above. Even inside its intended generated code, the branch for a dataset without `"train"` immediately accesses `dataset["train"]`, which contradicts the branch condition (`HyperTune/src/python/script_generator.py:306-317`).

The v2 file parses, but construction calls thirteen `_get_*_template()` methods that are never defined before EOF (`HyperTune/src/python/script_generator_v2.py:11-34`, `HyperTune/src/python/script_generator_v2.py:315-347`). Its format handlers therefore do not become reachable through a normal instance.

### 5.2 Non-Python outputs are not execution contracts — High confidence

- “Transformers” bash output invokes `python -m transformers.trainer` with custom arguments such as `--dataset_path` and `--batch_size`; the code supplies no executable module or argument adapter implementing that contract (`HyperTune/src/python/script_generator_v2.py:117-160`).
- DPO output similarly assumes a version-specific module path and CLI contract without validation (`HyperTune/src/python/script_generator_v2.py:164-194`).
- Model IDs, paths, and output directories are interpolated without shell quoting (`HyperTune/src/python/script_generator_v2.py:123-126`, `HyperTune/src/python/script_generator_v2.py:170-173`).
- JSON labels QLoRA as a PEFT type rather than LoRA plus a quantization configuration, and uses custom keys such as `lora_rank` instead of demonstrating a consumer schema (`HyperTune/src/python/script_generator_v2.py:249-266`).
- The hand-written YAML converter does not escape quoted scalar content and emits unquoted inline-list elements (`HyperTune/src/python/script_generator_v2.py:315-347`).

No test parses, imports, lints, or smoke-runs generated Python, JSON, YAML, or shell output. The integration suite explicitly substitutes a mock generator (`HyperTune/tests/integration/test_workflow.py:19-42`, `HyperTune/tests/integration/test_workflow.py:53-56`).

## 6. Real versus fake training and evaluation paths

### 6.1 Explicitly fake paths — High confidence

`tune_service.py` says it only defines an interface, then returns fixed `train_loss`, `val_loss`, and success values for LoRA, QLoRA, DoRA, AdaLoRA, and ReFT (`HyperTune/src/python/tune_service.py:9-20`, `HyperTune/src/python/tune_service.py:22-45`, `HyperTune/src/python/tune_service.py:47-72`, `HyperTune/src/python/tune_service.py:74-154`). These results must not be surfaced as training evidence.

The enhanced ReFT evaluator leaves task evaluation as `pass` and returns fixed accuracy/F1 of 0.9 (`HyperTune/src/python/reft_enhanced.py:348-412`).

### 6.2 Code that really calls training, but is not integrated or proven — Medium confidence

- The standalone LoRA trainer loads a causal LM, wraps it with PEFT LoRA, constructs a `Trainer`, calls `trainer.train()`, and saves outputs (`HyperTune/src/hypertuner/training/lora_trainer.py:26-47`, `HyperTune/src/hypertuner/training/lora_trainer.py:49-85`). This is a genuine training call, but it is isolated from both APIs and both generators, has no evaluation dataset, assumes a JSON `"text"` column, and hardcodes bfloat16.
- The generic trainer performs real forward, backward, optimizer, scheduler, and validation-loss steps (`HyperTune/src/python/train.py:79-109`, `HyperTune/src/python/train.py:115-180`). It does not install LoRA/QLoRA/DoRA. Its ReFT/hybrid setup is `pass` (`HyperTune/src/python/train.py:62-65`), so method names mainly select a checkpoint path rather than an implemented tuning algorithm (`HyperTune/src/python/train.py:35-50`).
- The PyReFT-style adapter constructs a third-party trainer and calls `train()` (`HyperTune/src/python/reft_adapter.py:194-258`). Its dataset preparation returns an essentially raw dataset without producing the intervention locations or tokenized fields expected by ReFT-style collators (`HyperTune/src/python/reft_adapter.py:139-188`); compatibility remains unproven.
- The vendored LoReFT script contains a substantial model/config/trainer path (`HyperTune/PyReft-Repo/loreft/train.py:263-353`), but imports an installed `pyreft` package and sibling modules by working-directory-relative names (`HyperTune/PyReft-Repo/loreft/train.py:28-45`). It is an upstream-style reference, not an integrated Aptus component.

### 6.3 Evaluation is weak even where it is real — High confidence

The standalone evaluator loads a base model and adapter and performs generation (`HyperTune/src/hypertuner/evaluation/lora_evaluator.py:19-59`). However:

- Perplexity is calculated on the input prompt only, not the expected output (`HyperTune/src/hypertuner/evaluation/lora_evaluator.py:65-71`).
- “Accuracy” is one stochastic sampled generation’s exact string equality (`HyperTune/src/hypertuner/evaluation/lora_evaluator.py:51-58`, `HyperTune/src/hypertuner/evaluation/lora_evaluator.py:73-77`).
- The evaluator is byte-identical in `training/` and `evaluation/` (`docs/audits/aptus-legacy/duplicate-clusters.json:104-111`).

It is real inference code, but not a credible quality benchmark without deterministic decoding, target-conditioned loss, task metrics, fixtures, and tests.

## 7. Advanced-method status

### 7.1 IA3, AdaLoRA, prompt/prefix tuning, and DPO — High confidence

The central optimizer can emit parameter dictionaries for these methods, but its selection is based only on fixed priors and formulas (`HyperTune/src/python/core_optimizer.py:475-599`, `HyperTune/src/python/core_optimizer.py:633-730`, `HyperTune/src/python/core_optimizer.py:773-892`). There is no integrated trainer for these configurations, and `tune_service.py` fakes AdaLoRA while not implementing its `spectral` or `mixture` CLI options (`HyperTune/src/python/tune_service.py:102-128`, `HyperTune/src/python/tune_service.py:156-189`).

### 7.2 Custom DoRA prototype — High confidence

The prototype stores magnitude and LoRA parameters in ordinary dictionaries rather than registered `nn.Module`/`ParameterDict` structures (`HyperTune/src/python/dora_decomposer.py:19-21`, `HyperTune/src/python/dora_decomposer.py:50-68`). It iterates parameter names ending in `.weight`, then assigns a Python function to that parameter attribute rather than replacing a module’s `forward` method (`HyperTune/src/python/dora_decomposer.py:37-41`, `HyperTune/src/python/dora_decomposer.py:70-109`). Finally it calls `save_pretrained()`, which does not serialize Python hooks/functions or unregistered parameter dictionaries (`HyperTune/src/python/dora_decomposer.py:111-129`). This is not a preservable trained DoRA model path.

### 7.3 Flexora prototype — High confidence

`apply_layer_weights()` writes detached `.item()` scalars to arbitrary `scale` attributes on standard attention/MLP modules (`HyperTune/src/python/flexora_optimizer.py:78-90`). The validation loss uses `self.alpha`, while the requested gradient is with respect to a separate clone `alpha_t` that was never used in that forward graph (`HyperTune/src/python/flexora_optimizer.py:124-144`). The advertised outer optimization is therefore disconnected from the loss and is expected to fail or produce no meaningful alpha update.

### 7.4 Custom ReFT hook prototype — High confidence

Intervention parameters are again stored in a plain dictionary (`HyperTune/src/python/reft_setup.py:7-15`, `HyperTune/src/python/reft_setup.py:86-104`). Runtime hooks are registered on the in-memory model (`HyperTune/src/python/reft_setup.py:106-146`), then `save_pretrained()` is used as though it persisted the hooks and unregistered parameters (`HyperTune/src/python/reft_setup.py:148-167`). The generic trainer does not reconstruct those hooks (`HyperTune/src/python/train.py:62-65`). This path cannot preserve the advertised intervention model.

### 7.5 Third-party ReFT material — Medium confidence

The external-library adapter and the vendored LoReFT training script are the only advanced-method paths worth adapting, but dependency and package boundaries are unresolved:

- The adapter requires an installed `pyreft` API (`HyperTune/src/python/reft_adapter.py:20-45`).
- The local `PyReft-Repo/pyreft/` tree has no `__init__.py`, and its dataset implementation is named `dataset.py.txt`, so it is not a complete importable local package (`docs/audits/aptus-legacy/inventory.jsonl:50-57`).
- The v2 stack pins Transformers 4.35.0 (`HyperTune/requirements_v2.txt:1-12`), while the vendored package requirements demand Transformers 4.48.2 or newer (`HyperTune/PyReft-Repo/pyreft/requirements.txt:5-16`).

Treat this material as a provenance/reference candidate until its source license, upstream revision, supported package versions, dataset contract, save/reload cycle, and tiny-model training test are established.

## 8. Test credibility

### 8.1 Collection is blocked — High confidence

Optimizer and integration tests import the parse-failed optimizer during collection (`HyperTune/tests/test_optimizer.py:7-17`, `HyperTune/tests/integration/test_workflow.py:8-17`). Parse-valid test files are therefore not passing tests; they are only syntactically valid test source.

### 8.2 Assertions mostly restate the implementation — High confidence

Optimizer tests check key presence, positivity, configured ranges, and explanation length (`HyperTune/tests/test_optimizer.py:19-49`). They do not compare recommendations with measured loss, quality, throughput, peak VRAM, convergence, or a baseline.

Some assertions contradict the formulas:

- A 30B unconstrained LoRA test expects batch size ≤ 8 (`HyperTune/tests/test_optimizer.py:88-99`), while the performance objective monotonically rewards the largest offered batch size, 16 (`HyperTune/src/python/core_optimizer.py:109-116`, `HyperTune/src/python/core_optimizer.py:130-145`).
- A 7B LoRA test supplies 12 GB and expects a fitting result (`HyperTune/tests/test_optimizer.py:100-115`), but base-model memory alone is estimated as 14 GB (`HyperTune/src/python/core_optimizer.py:911-915`).
- The macOS test expects `"16 GB"` parsing that the implementation does not support (`HyperTune/tests/test_resource_scanner.py:138-167`, `HyperTune/src/python/resource_scanner.py:153-167`).

### 8.3 “Integration” coverage substitutes the broken component — High confidence

The integration test says the generator is not implemented and uses `MockScriptGenerator` (`HyperTune/tests/integration/test_workflow.py:19-42`). It instantiates components directly rather than either REST app or the workflow class (`HyperTune/tests/integration/test_workflow.py:44-72`). It never trains or evaluates a model; “end to end” ends after writing a mock script (`HyperTune/tests/integration/test_workflow.py:79-130`).

The default test runner imports and schedules only optimizer test classes; it omits dataset, resource, integration, trainer, evaluator, generator, API, and advanced-method tests (`HyperTune/tests/run_tests.py:29-35`, `HyperTune/tests/run_tests.py:55-75`).

### 8.4 Duplication inflates apparent coverage — High confidence

Exact duplicate clusters include optimizer tests, dataset tests, resource tests, integration tests, integration fixtures/runners, and the main test runner (`docs/audits/aptus-legacy/duplicate-clusters.json:95-102`, `docs/audits/aptus-legacy/duplicate-clusters.json:248-264`, `docs/audits/aptus-legacy/duplicate-clusters.json:285-300`, `docs/audits/aptus-legacy/duplicate-clusters.json:333-339`). These copies add no independent evidence.

## 9. Dependency, package, and data gaps

### 9.1 No canonical environment — High confidence

The root requirements include the REST/optimizer basics but omit packages imported by core paths, including Transformers, Datasets, PEFT, bitsandbytes, Jinja2, TRL, and pyreft (`HyperTune/requirements.txt:1-32`). `requirements_v2.txt` adds several ML packages but still omits Jinja2, TRL, and pyreft (`HyperTune/requirements_v2.txt:1-12`). The source ReFT requirements introduce pyreft and a different broad Transformers constraint (`HyperTune/src/python/requirements.txt:1-15`), while the vendored requirements introduce pyvene and Transformers ≥ 4.48.2 (`HyperTune/PyReft-Repo/pyreft/requirements.txt:1-26`).

The four root requirement variants are explicitly grouped as one version family, not one authoritative lock (`docs/audits/aptus-legacy/version-families.json:99-107`).

### 9.2 Model data is missing or disconnected — High confidence

Both `ModelRegistry` and configuration point at a `Scrapers/data/base_models.json` path (`HyperTune/src/python/model_registry.py:16-34`, `HyperTune/src/python/config.py:17-22`), but no such file is present in the generated 228-file inventory (`docs/audits/aptus-legacy/baseline-manifest.json:4-15`). A manual file exists elsewhere and includes hosted/nonlocal model entries such as `gpt-4` and Claude models (`HyperTune/HyperTune-NEW_stuff_05-16-25/manual-models.json:1-44`), but it is not wired into the registry and is not a safe catalog of locally fine-tunable causal-LM checkpoints.

### 9.3 Filesystem and data contracts are deployment-specific — High confidence

Model, dataset, job, and export scripts hardcode global `/data/...` roots (`HyperTune/src/python/register_model.py:7-18`, `HyperTune/src/python/register_dataset.py:7-25`, `HyperTune/src/python/train.py:35-50`, `HyperTune/src/python/export_model.py:7-24`). These paths are not represented in API request contracts or a shared configuration object. Dataset contracts also vary among raw local files, Hugging Face dataset IDs, saved datasets, mandatory `"text"` columns, and DPO pairs without a common schema (`HyperTune/src/python/dataset_analyzer.py:151-197`, `HyperTune/src/hypertuner/training/lora_trainer.py:49-56`, `HyperTune/tests/integration/fixtures.py:158-188`).

## 10. Duplicate and version families

The generated baseline records 228 files, 38 exact-content clusters spanning 98 files, and 30 normalized version families (`docs/audits/aptus-legacy/baseline-manifest.json:2-15`). Python-relevant families include:

- Two exact root REST apps and two exact CLIs (`docs/audits/aptus-legacy/version-families.json:84-97`, `docs/audits/aptus-legacy/duplicate-clusters.json:49-56`, `docs/audits/aptus-legacy/duplicate-clusters.json:113-120`).
- Exact duplicate workflow classes and model registries in `src/python/` and `src/python 2/` (`docs/audits/aptus-legacy/version-families.json:149-163`, `docs/audits/aptus-legacy/duplicate-clusters.json:203-210`, `docs/audits/aptus-legacy/duplicate-clusters.json:350-357`).
- A three-member generator family containing one parse-failed v1 and two exact copies of the parse-valid-but-unconstructable v2 (`docs/audits/aptus-legacy/version-families.json:165-172`, `docs/audits/aptus-legacy/duplicate-clusters.json:58-65`).
- Exact duplicate test trees and runners (`docs/audits/aptus-legacy/version-families.json:175-244`).
- Exact duplicate evaluator files in semantically different directories (`docs/audits/aptus-legacy/duplicate-clusters.json:104-111`).
- Exact copied configuration and development-server files across dated and source directories (`docs/audits/aptus-legacy/duplicate-clusters.json:67-74`, `docs/audits/aptus-legacy/duplicate-clusters.json:341-348`).

This is not merely storage noise: divergent entry points and differently broken generator generations make ownership and behavior ambiguous.

## 11. KEEP / ADAPT / ARCHIVE / DISCARD hypotheses

These are disposition hypotheses, not migration actions.

### KEEP

- **Keep the generated forensic JSON evidence unchanged — High confidence.** It provides reproducible inventory, parse, family, and hash facts (`docs/audits/aptus-legacy/baseline-manifest.json:1-16`).
- **Keep the deterministic synthetic fixture-builder ideas — High confidence.** The integration fixture code creates local instruction, conversation, QA, and preference datasets without claiming model quality (`HyperTune/tests/integration/fixtures.py:7-30`, `HyperTune/tests/integration/fixtures.py:38-68`, `HyperTune/tests/integration/fixtures.py:158-190`).
- **No production Python runtime file should be kept unchanged — High confidence.** Every plausible runtime path requires at least contract, dependency, precision, validation, or integration changes documented above.

### ADAPT

- **Dataset analysis — Medium-high confidence.** Retain file-format loading and text/format extraction, but define one output schema, handle empty datasets and zero-token samples, seed sampling, and separate estimated from tokenizer-measured counts (`HyperTune/src/python/dataset_analyzer.py:41-149`, `HyperTune/src/python/dataset_analyzer.py:151-266`).
- **Resource discovery — Medium-high confidence.** Retain OS/GPU probes, but report per-device total/free memory, add Metal/unified-memory semantics, correct unit parsing and power-of-two rounding, and never convert aggregate VRAM into a single-device feasibility claim (`HyperTune/src/python/resource_scanner.py:30-55`, `HyperTune/src/python/resource_scanner.py:174-237`).
- **Standalone LoRA trainer/evaluator — Medium confidence.** Use as a minimal real-training starting point after adding configurable precision, schema-aware preprocessing, validation, deterministic evaluation, measured peak memory/throughput, and API integration (`HyperTune/src/hypertuner/training/lora_trainer.py:23-85`, `HyperTune/src/hypertuner/evaluation/lora_evaluator.py:16-88`).
- **Third-party LoReFT path — Medium confidence.** Adapt from a verified upstream revision/package rather than the incomplete local package tree; prove dataset preparation and save/reload behavior with a tiny model (`HyperTune/PyReft-Repo/loreft/train.py:28-45`, `HyperTune/PyReft-Repo/loreft/train.py:263-353`).
- **Model/dataset registration concepts — Medium confidence.** Make storage roots configurable, validate formats and IDs, record revisions/checksums, and return typed failures instead of embedding deployment assumptions (`HyperTune/src/python/register_model.py:7-44`, `HyperTune/src/python/register_dataset.py:7-51`).

### ARCHIVE

- **Archive all exact “2” copies and dated copies after choosing a canonical reference — High confidence.** Hash evidence proves they add no distinct implementation (`docs/audits/aptus-legacy/duplicate-clusters.json:49-74`, `docs/audits/aptus-legacy/duplicate-clusters.json:248-300`, `docs/audits/aptus-legacy/duplicate-clusters.json:333-357`).
- **Archive the three incompatible orchestration surfaces as design history — High confidence.** Their request schemas and component contracts have diverged too far to serve as a production base (`HyperTune/hyperparameter_mcp.py:43-129`, `HyperTune/api/FastAPI/main.py:27-65`, `HyperTune/src/python/hyperparameter_mcp.py:62-202`).
- **Archive vendored ReFT/original-code snapshots pending provenance — Medium confidence.** They may be useful upstream references but are not a coherent local package (`docs/audits/aptus-legacy/inventory.jsonl:37-57`).

### DISCARD

- **Discard the mislabeled `.py` guide copy — High confidence.** It is prose and byte-identical to Markdown (`HyperTune/Complete Guide to Building & Deploying.py:1-10`, `docs/audits/aptus-legacy/duplicate-clusters.json:77-84`).
- **Discard fixed-metric training/evaluation shims as executable product code — High confidence.** Keeping them risks reporting fabricated success (`HyperTune/src/python/tune_service.py:22-154`, `HyperTune/src/python/reft_enhanced.py:398-412`).
- **Discard the current central optimizer implementation as a production engine — High confidence.** Its useful search-space priors can be transcribed into documentation, but the implementation is parse-failed, nonempirical, dimensionally unsafe, and semantically overclaims its output (`HyperTune/src/python/core_optimizer.py:97-149`, `HyperTune/src/python/core_optimizer.py:911-949`, `docs/audits/aptus-legacy/reference-map.json:3633-3642`).
- **Discard both script-generator implementations and rebuild from validated templates — High confidence.** One does not parse; the other cannot construct and emits unverified command contracts (`docs/audits/aptus-legacy/reference-map.json:4512-4521`, `HyperTune/src/python/script_generator_v2.py:11-41`, `HyperTune/src/python/script_generator_v2.py:117-218`).
- **Discard the custom DoRA, Flexora, and hook-only ReFT prototypes as runtime implementations — High confidence.** Their trainable state, computation graph, or persistence mechanism is structurally broken (`HyperTune/src/python/dora_decomposer.py:37-109`, `HyperTune/src/python/flexora_optimizer.py:124-150`, `HyperTune/src/python/reft_setup.py:86-165`).

## 12. Minimum evidence required before reuse

A future Aptus Python backend should not claim functionality until all of these are demonstrated:

1. One canonical installable package, one dependency lock, one API schema, and one entry point.
2. Parse/compile success for every shipped Python file and every generated Python script.
3. Import and construction tests that do not replace the component under test.
4. A tiny local end-to-end run from request → dataset analysis → recommendation → generated/selected trainer → training → deterministic evaluation → save → reload.
5. Empirical optimization in which each trial receives a measured validation objective, or honest renaming to “heuristic recommendation.”
6. Peak-memory and throughput measurements calibrated by device, model revision, dtype, quantization backend, sequence length, batch size, and sharding strategy.
7. Advanced-method tests that prove parameters are registered, receive gradients, change after an optimizer step, serialize, reload, and reproduce outputs.
8. Test reports and artifacts that distinguish mocked unit coverage from real CPU/GPU integration coverage.
