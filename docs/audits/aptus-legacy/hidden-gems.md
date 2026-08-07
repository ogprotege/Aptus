# Aptus Legacy Hidden Gems

> **Documentation status:** Archived evidence
>
> **Applies to:** Dated forensic review of the removed legacy `HyperTune/` tree
>
> **Last reviewed:** 2026-08-06
>
> **Next scheduled review:** 2027-08-06, or when provenance or reproduction paths change
>
> **Historical warning:** The body below is preserved point-in-time evidence,
> not current product or implementation guidance. Start with the
> [audit index](README.md) or [current capabilities](../../product/current-capabilities.md).

## Forensic recovery report

This report identifies ideas and code seams worth recovering for Aptus from the legacy source tree. `HyperTune/` is used below only as the legacy source path, not as the product name.

This ranking began with read-only static inspection. Later disposable checks are summarized in `sandbox-summary.md`: no training job or generated artifact ran, but bounded checks corroborated the broken builds/generators and exercised the resource scanner with a dependency shim. “Parse-passed” still means only syntax acceptance unless separate sandbox evidence is cited.

### Confidence labels

- **High** — directly established by source text, hashes, duplicate evidence, missing-import evidence, or recorded parser output.
- **Medium** — a coherent seam is present, but runtime behavior, data quality, or scientific validity has not been established.
- **Low** — a prose claim, unexplained constant, inferred compatibility label, or heuristic without traceable validation.

## Executive verdict

The archive contains a valuable product thesis but not a trustworthy tuning system. Aptus should recover the workflow contracts, model/task metadata shapes, architecture-specific adapter hints, research provenance, and explanation surfaces. It should not carry forward the legacy formulas, claimed “optimal” values, compatibility labels, memory estimates, success metrics, or generated commands as production logic.

The highest-value differentiator is also the largest missing component: a **calibrated resource, quantization, and precision planner coupled to generated-script validation**. The archive gestures toward both halves but implements neither reliably.

The generated inventory supports this conservative posture:

- 228 files were inventoried: 201 text, 4 binary, and 23 empty; 38 exact-duplicate clusters cover 98 files, and 30 version families were detected (`docs/audits/aptus-legacy/baseline-manifest.json:2-15`).
- Important duplicates include the two copies of `script_generator_v2.py` and a TypeScript file that is byte-identical to a prose/code dump (`docs/audits/aptus-legacy/duplicate-clusters.json:59-65`, `docs/audits/aptus-legacy/duplicate-clusters.json:132-138`).
- Empty files include claimed method, model, output, API, and test surfaces (`docs/audits/aptus-legacy/inventory.jsonl:70-84`, `docs/audits/aptus-legacy/inventory.jsonl:114-122`, `docs/audits/aptus-legacy/inventory.jsonl:163-181`).
- The reference map records 40 missing local import edges. Examples include every method imported by the central method factory and the estimator/output modules imported by the TypeScript optimizer (`docs/audits/aptus-legacy/reference-map.json:3070-3131`, `docs/audits/aptus-legacy/reference-map.json:3307-3366`).
- Three Python files fail the recorded syntax parse, including the main Python optimizer and original script generator (`docs/audits/aptus-legacy/reference-map.json:3633-3642`, `docs/audits/aptus-legacy/reference-map.json:4512-4521`).
- TypeScript files were not parser- or type-check-validated by the generated evidence; their `parse_status` is `not_checked` (`docs/audits/aptus-legacy/reference-map.json:1913-2055`).

## Ranked recovery candidates

### 1. Feasibility-first resource, quantization, and precision planner

- **Category:** Product workflow idea plus partial code seam.
- **Sources:** `HyperTune/src/python/resource_scanner.py:15-55`, `HyperTune/src/python/resource_scanner.py:174-237`, `HyperTune/src/python/core_optimizer.py:9-28`, `HyperTune/src/python/core_optimizer.py:211-328`, `HyperTune/src/hypertuner/methodSelector.ts:4-20`, `HyperTune/src/hypertuner/methodSelector.ts:22-111`, `HyperTune/src/output/config_file.ts:29-43`.
- **Why it matters:** Fine-tuning users first need to know what can run safely, on which hardware, at what precision, with what quantization and batch strategy. A recommendation that cannot fit is worse than no recommendation. A calibrated feasibility planner can become Aptus’s defensible core.
- **What is actually present:** A cross-platform resource probe; model size, sequence length, dataset size, GPU-memory, and user-priority inputs; rough LoRA/QLoRA/full-tuning memory branches; NF4/FP4 and double-quantization choices; and a CUDA BF16 capability check.
- **Operational status:** **Not operational as a planner.** The scanner is syntax-parse-passed (`docs/audits/aptus-legacy/reference-map.json:4470-4510`). Its as-is smoke check stopped at an unresolved `psutil` dependency; a disposable shim then completed the scanner's basic output contract. This supports adaptation of the discovery seam, not its recommendations. Its batch-size calculation is explicitly rough, sums all GPU memory as if uniformly usable, and its “round down to a power of two” expression can return one less than a power of two (`HyperTune/src/python/resource_scanner.py:179-191`, `HyperTune/src/python/resource_scanner.py:211-237`). The main Python optimizer fails parsing at line 1021 (`docs/audits/aptus-legacy/reference-map.json:3633-3642`). Its search objective scores hand-written proxies rather than observed training outcomes, and its memory arithmetic omits or compresses major terms (`HyperTune/src/python/core_optimizer.py:118-149`, `HyperTune/src/python/core_optimizer.py:236-275`).
- **Confidence/provenance:** **High** that the product seam and fragments exist; **low** that any estimate is accurate. Constants and formulas have no calibration dataset, error bounds, hardware/version key, or benchmark provenance.
- **Validate or rewrite:** Keep the input/output contract; rewrite the estimator. Model weights, gradients, optimizer states, adapters, quantization metadata, activations, attention, temporary kernels, allocator fragmentation, checkpointing, sequence-length distribution, packing, per-device batch, accumulation, sharding/offload, and safety margin separately. Detect CUDA/ROCm/MPS/backend capabilities and distinguish aggregate memory from per-device feasibility. Choose BF16, FP16, FP32, 8-bit, or 4-bit only when the hardware, framework, model, and training method support the combination. Calibrate predictions against versioned dry-run and real-job telemetry, return confidence intervals and assumptions, and fail closed when evidence is insufficient.

### 2. Typed generation pipeline with mandatory artifact validation

- **Category:** Product workflow idea plus incomplete generator seam.
- **Sources:** `HyperTune/src/python/script_generator_v2.py:11-42`, `HyperTune/src/python/script_generator_v2.py:44-101`, `HyperTune/src/python/script_generator_v2.py:222-347`, `HyperTune/src/python/script_generator.py:178-210`, `HyperTune/src/python/script_generator.py:545-578`, `HyperTune/src/output-generators.ts:3-65`, `HyperTune/tests/integration/test_workflow.py:19-35`.
- **Why it matters:** Translating one verified plan into framework-specific, reproducible artifacts is useful by itself. Validation before users spend GPU time is the natural companion to the resource planner and a second strong Aptus differentiator.
- **What is actually present:** A useful dispatch shape covering framework × method × output format, plus generic JSON/YAML serialization and fragments for Transformers, LLaMA Factory, Axolotl, LoRA, QLoRA, full tuning, PEFT, and DPO.
- **Operational status:** **Broken/incomplete.** The original generator has a recorded `SyntaxError` (`docs/audits/aptus-legacy/reference-map.json:4512-4521`), contains `pass` branches for advertised methods/frameworks, and ends in an incomplete generated shell block (`HyperTune/src/python/script_generator.py:178-210`, `HyperTune/src/python/script_generator.py:545-578`). The v2 file parse-passes, but its constructor calls template-provider methods that are never defined before the file ends at line 347 (`HyperTune/src/python/script_generator_v2.py:11-33`, `HyperTune/src/python/script_generator_v2.py:103-347`). Integration tests explicitly substitute a mock because the generator is “not implemented yet” (`HyperTune/tests/integration/test_workflow.py:19-24`). Other generators emit guessed module entry points and fixed training defaults without evidence that those CLIs exist (`HyperTune/src/output-generators.ts:3-30`, `HyperTune/src/output/command_line.ts:30-95`).
- **Confidence/provenance:** **High** for the abstraction and failure status; **low** for generated artifact correctness. The v2 file is also duplicated byte-for-byte (`docs/audits/aptus-legacy/duplicate-clusters.json:59-65`).
- **Validate or rewrite:** Introduce a typed, versioned intermediate `TrainingPlan`; make each framework adapter consume it; quote paths and secrets safely; pin supported framework-version ranges; and prohibit generated install commands by default. Every output must pass: plan-schema validation, framework-config schema validation, Python AST or shell/YAML/JSON/TOML parsing, import/entry-point and CLI-flag checks, model target-module verification, dataset-contract checks, a no-download/no-training static mode, and an optional bounded one-batch dry run with peak-memory capture. A generated artifact is “ready” only when its validation report is attached.

### 3. Model capability registry backed by live config introspection

- **Category:** Trustworthy data-contract seam with untrustworthy records.
- **Sources:** `HyperTune/src/model-database.ts:1-22`, `HyperTune/src/model-database.ts:24-335`, `HyperTune/src/model-database.ts:337-381`, `HyperTune/src/model-database-update.ts:339-400`, `HyperTune/src/model-database-update.ts:403-491`, `HyperTune/src/models/index.ts:1-66`.
- **Why it matters:** Planning depends on normalized architecture, parameter count, hidden size, layer/head counts, context limits, training support, quantization support, license, and source identity. The schema is a useful foundation for Aptus’s planner and generator.
- **What is actually present:** Two competing registries, a reasonably rich `ModelInfo` shape, exact/family lookup helpers, fuzzy name detection, a merge function, compatibility labels, and hard-coded records for several model families.
- **Operational status:** **Data-only and internally split.** The direct database can be read as static data, but the separate registry imports empty `llama.ts` and `gemma.ts` files (`HyperTune/src/models/index.ts:58-66`; `docs/audits/aptus-legacy/inventory.jsonl:169-171`). The config-reading function explicitly returns `null` instead of reading a config (`HyperTune/src/model-database-update.ts:476-491`). Fuzzy substring matching can silently select the wrong model (`HyperTune/src/model-database.ts:358-381`). Several architecture sizes and `bestMethods` entries are asserted without field-level sources or benchmark scope.
- **Confidence/provenance:** **Medium** for the schema and pure lookup concept; **low** for the static records and method-compatibility labels. The inventory supplies hashes and timestamps, not scientific provenance (`docs/audits/aptus-legacy/inventory.jsonl:167-168`).
- **Validate or rewrite:** Establish one canonical registry keyed by immutable provider/model/revision identity. Populate structural fields from the actual model config or a versioned trusted source, record provenance and retrieval time per field, and distinguish declared, inferred, measured, and user-overridden values. Replace fuzzy auto-selection with ranked candidates plus confirmation. Derive target modules from inspected module names. Treat license and training permission as first-class gates. Never infer method support solely from model size.

### 4. Provenance-aware task and method benchmark catalog

- **Category:** Research reference plus promising heuristic.
- **Sources:** `HyperTune/src/hypertuner/task-configs.ts:25-359`, `HyperTune/src/hypertuner/task-configs.ts:361-405`, `HyperTune/docs/reft_methods_guide.md:98-170`, `HyperTune/docs/reft_methods_guide.md:229-231`, `HyperTune/PyReft-Repo/loreft/README.md:3-7`, `HyperTune/PyReft-Repo/loreft/README.md:29-122`.
- **Why it matters:** Aptus can provide much better starting points when recommendations are scoped to a model revision, task, dataset, method implementation, hardware/software stack, metric, and observed result rather than labeled universally “optimal.”
- **What is actually present:** A broad task × method matrix for commonsense reasoning, arithmetic reasoning, instruction tuning, classification, summarization, and general use, including rank, target layers/modules, learning rate, decay, batch, warmup, and epochs. ReFT documentation cites arXiv:2404.03592v3, and the copied example README links to StanfordNLP/PyReFT and exposes concrete commands, seeds, datasets, and evaluation constraints.
- **Operational status:** **Useful research seed, unsafe default source.** Some ReFT values resemble the copied commands, but many layer, position, warmup, epoch, and decay values do not match those commands exactly. The TypeScript getter mutates the shared selected configuration before returning a copy, so repeated model-specific calls can compound changes (`HyperTune/src/hypertuner/task-configs.ts:367-405`). Its method-comparison metrics are literal placeholders (`HyperTune/src/hypertuner/task-configs.ts:442-463`).
- **Confidence/provenance:** **Medium** for the traceable ReFT research lead; **low** for the normalized preset table as a whole. Only ReFT has a full local citation. The copied PyReFT subtree has no included license file, while the legacy root’s license is only the text “MIT” without a complete license grant (`HyperTune/LICENSE:1-2`); reuse requires provenance and licensing review.
- **Validate or rewrite:** Store presets as immutable benchmark observations, not global defaults. Require source URL/commit or paper/table, model revision, dataset split, seed count, metric, framework/library versions, hardware, exact command, and result distribution. Reproduce a small accepted matrix before promoting any prior. Adapt layer indices to architecture only through verified mappings. Return “research prior” and uncertainty, never “optimal,” until Aptus has measured evidence.

### 5. Architecture-aware target-module resolver

- **Category:** Promising heuristic and code seam.
- **Sources:** `HyperTune/src/formulas/target_modules.ts:14-132`, `HyperTune/src/formulas/target_modules.ts:135-200`, `HyperTune/src/python/core_optimizer.py:42-63`.
- **Why it matters:** Correct adapter attachment is a common source of silent under-training or immediate failure. Minimal, balanced, and comprehensive target sets are also a useful capacity/resource control.
- **What is actually present:** A readable architecture map for Llama-family, Mistral, Mixtral, Falcon, Gemma, Phi, Qwen, GPT-NeoX, BERT, RoBERTa, and DeBERTa module names, with three target intensities and a resource/goal selector.
- **Operational status:** **Not production-safe.** Unknown models silently default to Llama modules (`HyperTune/src/formulas/target_modules.ts:151-168`). Resource thresholds automatically broaden targets without estimating the resulting trainable parameters or memory (`HyperTune/src/formulas/target_modules.ts:170-191`). `MethodType` is referenced but not defined or imported in this file, and TypeScript parsing/type checking was not performed by the audit.
- **Confidence/provenance:** **Medium** that many names are useful fallback hints; **low** that every family/version and intensity policy is correct. No model revision or source accompanies the map.
- **Validate or rewrite:** Introspect the loaded architecture, enumerate candidate linear modules, verify every selected module exists, calculate trainable parameter count and memory delta, and require explicit handling for fused projections, MoE experts/routers, encoder-decoder stacks, and remote custom code. Keep the legacy map only as a tested alias library. Unknown architectures must return “needs inspection,” not a Llama default.

### 6. Explainable method-constraint and recommendation engine

- **Category:** Product workflow idea plus low-confidence heuristic.
- **Sources:** `HyperTune/src/method-constraints.ts:2-59`, `HyperTune/src/hypertuner/methodSelector.ts:14-67`, `HyperTune/src/hypertuner/methodSelector.ts:70-111`, `HyperTune/src/methods.ts:43-105`.
- **Why it matters:** Aptus should explain why a method is feasible, rejected, or preferred and show tradeoffs among memory, speed, quality risk, trainable parameters, and inference behavior.
- **What is actually present:** A compact method-constraint schema; a recommendation result carrying configuration, estimated memory/time/accuracy; priority-based branching; and a unified workflow concept that detects a model, checks compatibility, generates a configuration, merges user overrides, and dispatches a method.
- **Operational status:** **Conceptual/broken.** The central method factory imports seven missing method modules (`docs/audits/aptus-legacy/reference-map.json:3070-3131`). The selector’s accuracy values are fixed at 0.85, 0.95, and 0.90, while memory/time formulas are unexplained arithmetic (`HyperTune/src/hypertuner/methodSelector.ts:22-67`, `HyperTune/src/hypertuner/methodSelector.ts:82-111`). Unknown constraints silently fall back to LoRA (`HyperTune/src/method-constraints.ts:56-58`).
- **Confidence/provenance:** **High** for the desired decision contract; **low** for all scores, savings, compatibility, and performance estimates.
- **Validate or rewrite:** Build a fail-closed rule layer over the calibrated planner and benchmark catalog. Separate hard constraints from preferences, emit rejection reasons and alternatives, expose assumptions and confidence, and rank feasible plans on a Pareto frontier. Unknown methods/models must produce an explicit unsupported result. User overrides must be revalidated after merge.

### 7. End-to-end “analyze → plan → explain → generate → validate” workflow

- **Category:** Product workflow idea.
- **Sources:** `HyperTune/SystemArchitecture.md:1-16`, `HyperTune/README.md:65-137`, `HyperTune/Random/script_generator_info.md:43-52`, `HyperTune/Integration_Architecture.md:1-21`.
- **Why it matters:** The archive’s best product insight is a guided handoff from model, dataset, task, and hardware evidence to a reviewable plan and deployable artifact. Aptus can reduce integration work without pretending to replace empirical tuning.
- **What is actually present:** Architecture diagrams connect an API, optimization core, explanation engine, generator, and output formatter. Documentation describes multiple frameworks and formats and an end-to-end sequence from dataset/resource analysis to script output.
- **Operational status:** **Architecture/prose only as a complete workflow.** The small system diagram is a useful boundary sketch, but the broader integration diagram lists many providers and platforms without corresponding implementations. The reference map records numerous missing provider, framework, and deployment modules (`docs/audits/aptus-legacy/reference-map.json:2730-2882`).
- **Confidence/provenance:** **High** that this was the intended workflow; **low** that any advertised integration worked.
- **Validate or rewrite:** Narrow the first Aptus workflow to supported, version-pinned adapters. Make the plan and validation report durable artifacts. Preserve human review and explicit overrides. Add execution only after planning and generation are independently trustworthy.

### 8. Forensic canonicalization and provenance gate

- **Category:** Trustworthy evidence/tooling seam.
- **Sources:** `docs/audits/aptus-legacy/baseline-manifest.json:2-15`, `docs/audits/aptus-legacy/duplicate-clusters.json:58-65`, `docs/audits/aptus-legacy/duplicate-clusters.json:303-330`, `docs/audits/aptus-legacy/version-families.json:109-171`, `docs/audits/aptus-legacy/reference-map.json:2382-2408`.
- **Why it matters:** Aptus should migrate one evidence-backed canonical source per concept, not accidentally select an older copy, prose paste, empty stub, or file with missing imports.
- **What is actually present:** Content hashes, file kinds/sizes, duplicate clusters, normalized version families, import resolution, and Python parser results. The secret scan records no findings in its scanned scope (`docs/audits/aptus-legacy/secret-scan.json:1-4`).
- **Operational status:** **Usable as static forensic evidence.** It does not establish runtime correctness, dependency compatibility, scientific validity, or complete secret safety.
- **Confidence/provenance:** **High** for the recorded snapshot and exact duplicates; bounded by the manifest timestamp, source root, scanner scope, and the fact that TypeScript parsing was not checked.
- **Validate or rewrite:** Preserve hashes and source lineage for every recovered fragment. Select canonical representatives before porting. Require fresh language parsing, type checking, dependency resolution, tests, licenses, and benchmark provenance in the new Aptus tree. Do not treat the zero-finding secret scan as a general security review.

## Evidence classes

### Trustworthy code and evidence

No end-to-end legacy tuning path qualifies as trustworthy or production-ready.

The narrowly trustworthy/reusable material is:

1. **The generated forensic snapshot** for file existence, emptiness, hashes, duplicates, version families, import edges, and recorded Python parse results.
2. **The model and recommendation data shapes** as design starting points, especially `ModelInfo`, `ModelMetrics`, `MethodRecommendation`, and `OptimizationContext` (`HyperTune/src/model-database.ts:1-22`, `HyperTune/src/hypertuner/methodSelector.ts:4-20`, `HyperTune/src/python/core_optimizer.py:9-28`). Their records and estimates are not trustworthy.
3. **The resource enumeration seam** as parse-confirmed, inspectable code for CPU/RAM and several GPU discovery paths (`HyperTune/src/python/resource_scanner.py:15-172`). Its usable-memory and batch recommendations are heuristics, not trustworthy planning.
4. **The static model lookup/filter helpers** as simple implementation ideas (`HyperTune/src/model-database.ts:337-355`). Fuzzy detection and the underlying records must be replaced or verified.

### Promising heuristics

These are useful hypotheses for experiments, not defaults:

- Architecture-specific adapter target names and minimal/balanced/comprehensive target sets (`HyperTune/src/formulas/target_modules.ts:14-132`).
- Task × method prior configurations, especially those traceable to ReFT experiments (`HyperTune/src/hypertuner/task-configs.ts:25-359`).
- Bounded warmup and training-schedule policies (`HyperTune/src/formulas/warmup.ts:16-77`, `HyperTune/src/formulas/training_steps.ts:18-98`).
- Method constraint fields such as rank range, training speed class, inference overhead, and model support (`HyperTune/src/method-constraints.ts:3-53`).

The formula collection must not be ported as a coherent optimizer:

- Two rank variants disagree on DoRA direction: one reduces rank to 0.8× while the other increases it to 1.25× (`HyperTune/src/formulas/rank.ts:26-44`, `HyperTune/src/formulas/rank_v2.ts:24-41`).
- Two weight-decay variants encode materially different baselines and multipliers (`HyperTune/src/formulas/weight_decay.ts:11-63`, `HyperTune/src/formulas/weight_decay_v2.ts:13-70`).
- One learning-rate function adds nondeterministic random variation to an allegedly optimal result (`HyperTune/src/formulas/learning_rate.ts:60-64`).
- Another claims a scaling-law basis without citation and multiplies the base rate by 10× or 12× before a broad cap (`HyperTune/src/formulas/learning_rate_2.ts:15-27`, `HyperTune/src/formulas/learning_rate_2.ts:100-101`).
- The formula index exports names the implementation files do not export and calls them with incompatible object-shaped arguments (`HyperTune/src/formulas/index.ts:4-11`, `HyperTune/src/formulas/index.ts:35-67`; compare `HyperTune/src/formulas/learning_rate.ts:6-10`, `HyperTune/src/formulas/batch_size.ts:6-10`).
- `learning_parameters.ts` is an exact duplicate of a file under `Random/`, indicating a pasted bundle rather than a maintained source module (`docs/audits/aptus-legacy/duplicate-clusters.json:132-138`).

### Research references

1. **ReFT / LoReFT / DiReFT:** The strongest traceable lead. The guide cites Wu et al., arXiv:2404.03592v3 (`HyperTune/docs/reft_methods_guide.md:229-231`). The copied README identifies StanfordNLP/PyReFT, seeds, datasets, commands, evaluation constraints, and logged runs (`HyperTune/PyReft-Repo/loreft/README.md:3-7`, `HyperTune/PyReft-Repo/loreft/README.md:29-122`). Treat this as research material until the exact upstream revision, license, and reproducibility are verified.
2. **LoftQ:** A URL is attached to generation defaults in the copied task configuration (`HyperTune/PyReft-Repo/loreft/task_config.py:137-155`). This is a lead, not provenance for the legacy optimizer.
3. **DoRA and AdaLoRA:** They are repeatedly named, but the inspected architecture/formula documentation supplies no complete local bibliography tying specific constants to paper tables or code revisions. Their legacy values are unsupported until traced independently.

Research results must retain benchmark scope. A paper’s result for a particular model, dataset, seed set, and implementation cannot justify a universal Aptus compatibility or quality claim.

### Product workflow ideas

- Ask for model/revision, task, dataset statistics, sequence-length distribution, hardware, budget, and quality/speed/memory priorities.
- Return several feasible plans, not one unexplained “optimal” answer.
- Show hard constraints, assumptions, predicted resource envelope, confidence, tradeoffs, and rejected alternatives.
- Convert the selected plan into typed, framework-versioned artifacts.
- Validate every artifact and attach the report before execution.
- Record observed peak memory, throughput, loss behavior, failures, and environment versions to improve calibration.
- Produce a reproducibility bundle: plan, source provenance, generated artifacts, validation results, environment lock, and run metadata.

These ideas follow the intended boundaries in `HyperTune/SystemArchitecture.md:1-16`, but Aptus should implement them as a measured compiler/planner workflow rather than a formula-driven oracle.

### Unsupported or fabricated claims

The following must not be repeated as Aptus capabilities:

- **Fake successful training:** `tune_service.py` returns `success: True`, fixed losses, fixed “75%” memory savings, and other metrics while comments state the real implementation is absent (`HyperTune/src/python/tune_service.py:22-45`, `HyperTune/src/python/tune_service.py:47-100`, `HyperTune/src/python/tune_service.py:102-154`).
- **Placeholder comparisons presented as metrics:** parameter efficiency, training speed, inference speed, and expected performance are hard-coded to 0.95, 0.8, 0.9, and 0.85 (`HyperTune/src/hypertuner/task-configs.ts:442-463`).
- **Fabricated accuracy impacts:** method recommendations use fixed relative values of 0.85, 0.95, and 0.90 without benchmark evidence (`HyperTune/src/hypertuner/methodSelector.ts:22-67`).
- **Unproven savings and compatibility:** method constraints assert fixed VRAM savings and wildcard/model-family support without sources (`HyperTune/src/method-constraints.ts:3-53`).
- **Advertised generator/framework support:** prose claims Transformers, LLaMA Factory, Axolotl, and multiple output formats, while another document admits the full template methods still need to be added (`HyperTune/Random/script_generator_info.md:1-14`, `HyperTune/Random/script_generator_info.md:14-22`).
- **“Currently supports” ReFT:** the guide claims support (`HyperTune/docs/reft_methods_guide.md:9-15`), but the implementation guide refers to a missing TypeScript method file and instructs readers how to integrate it (`HyperTune/docs/reft_implementation_guide.md:7-23`, `docs/audits/aptus-legacy/reference-map.json:2678-2685`).
- **Universal “optimal” and “eliminates guesswork” language:** the roadmap itself says the formulas, comprehensive model database, server, generators, and real-world tests are still next steps (`HyperTune/options-v1.md:29-51`). These are aspirations, not evidence.
- **Broad integration coverage:** the integration diagram is a product map, not implemented provider support (`HyperTune/Integration_Architecture.md:1-21`; `docs/audits/aptus-legacy/reference-map.json:2730-2882`).
- **Unscoped ReFT performance language:** “15–65x” and “better performance” may summarize research, but the local guide does not preserve the benchmark denominator and scope at the claim site (`HyperTune/docs/reft_methods_guide.md:13-27`). Aptus must quote the underlying experiment precisely or omit the claim.

## Required Aptus differentiator

### A. Calibrated resource, quantization, and precision planner

Aptus should produce a versioned `TrainingPlan` with:

- exact model identity/revision and inspected architecture facts;
- hardware/backend inventory per device, including usable per-device memory and supported dtypes/kernels;
- dataset token and sequence-length distributions, packing policy, and evaluation split;
- method, target modules/layers, trainable parameter count, rank/alpha/dropout;
- weight/storage dtype, compute dtype, quantization type, double quantization, optimizer dtype/state format, and compatibility rationale;
- micro-batch, gradient accumulation, effective batch, checkpointing, attention backend, offload/sharding, and distributed strategy;
- predicted base, adapter, gradient, optimizer, activation, temporary, and fragmentation memory;
- expected peak VRAM as a range, safety margin, throughput/time range, assumptions, calibration cohort/version, and confidence;
- feasible alternatives and explicit rejection reasons.

Calibration must come from measured dry runs and real jobs keyed by GPU, backend, framework/library versions, model architecture/size, method, dtype/quantization, sequence length, batch, checkpointing, attention backend, and optimizer. Aptus should continuously compare predicted versus observed peak memory and throughput, retain residual distributions, and widen uncertainty or abstain outside calibrated regions.

Minimum acceptance criteria:

1. A plan never claims aggregate multi-GPU memory is usable without a supported sharding strategy.
2. Unsupported precision/quantization/backend combinations are rejected before generation.
3. Predicted peak memory includes a reported safety margin and confidence interval.
4. The planner returns “insufficient evidence” instead of silently defaulting.
5. “Recommended” always means “under these stated assumptions,” never universally optimal.

### B. Generated-script validation

Validation should be a first-class Aptus artifact, not an optional test:

1. **Plan validation:** typed schema, ranges, cross-field invariants, user-override revalidation, and provenance completeness.
2. **Static artifact validation:** Python AST; JSON/YAML/TOML schema; shell parsing and quoting; forbidden install/secret interpolation checks.
3. **Dependency validation:** supported package-version matrix, imports, entry points, CLI flags, and deprecation checks.
4. **Model validation:** resolved model revision, task/model class, tokenizer/padding/chat template, target module existence, dtype/quantization support, and trainable parameter count.
5. **Dataset validation:** source resolution, schema/column mapping, split availability, preference-pair requirements, tokenization, labels/masking, truncation, and a small sample transform.
6. **Bounded runtime validation:** construct model/config without training when possible; otherwise run one forward/backward/optimizer step under a strict budget, capture peak memory, and compare it with the plan.
7. **Artifact validation:** output path, checkpoints, adapter/config/tokenizer files, resume behavior, deterministic seed capture, and a machine-readable report.

The validator must distinguish `invalid`, `unsupported`, `static-pass`, `dry-run-pass`, and `measured-run-pass`. A syntax-valid template is not an executable training script.

## Carry-forward decision

### Adopt as interfaces

- Resource/model/dataset/task input contracts.
- A versioned training-plan intermediate representation.
- Model capability and method-constraint schemas.
- Multiple feasible recommendations with explanations.
- Framework adapter and output-format dispatch.
- Validation and reproducibility reports.
- Forensic source hashes and provenance.

### Adapt only after measurement

- Target-module alias maps.
- Task/method presets.
- Rank, learning-rate, decay, warmup, epoch, and batch priors.
- ReFT configuration guidance.
- Memory, throughput, and time estimators.
- Method scoring and compatibility rules.

### Research-reference only

- Vendored PyReFT material until upstream revision and licensing are established.
- ReFT/LoftQ commands and defaults until reproduced.
- DoRA/AdaLoRA prose until constants are traced to primary sources.

### Reject as implementation

- Fake success/metric paths.
- Placeholder accuracy/performance numbers.
- Silent LoRA or Llama fallbacks.
- Static “best method” labels without evidence.
- The current formula bundle and optimizer objective.
- Both current script generators as production code.
- Generated commands that install dependencies or target guessed module entry points.
- Any claim that the legacy tree already provides working end-to-end tuning, broad integrations, or validated optimal configurations.

## Bottom line

Aptus has a credible opportunity to turn the archive’s best idea into a rigorous product: inspect the real workload and hardware, generate several evidence-scoped feasible plans, explain the tradeoffs, compile the selected plan into framework-specific artifacts, validate those artifacts, and learn from measured runs. The legacy archive contributes useful contracts, maps, research leads, and failure lessons. The planner, scientific calibration, generator adapters, and validation system must be rewritten.
