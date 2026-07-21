# Aptus Legacy Failure and Risk Register

## Decision

**Required disposition: quarantine the legacy `HyperTune/` folder as reference-only evidence. Do not build, deploy, publish, license, sell, or import it into Aptus until every Critical and High item below has an owner, authoritative source evidence, and a verified replacement or explicit rejection decision.**

The audited material is not a coherent, deployable product. It combines incomplete implementations, generated prose saved as source, empty commercial scaffolding, conflicting deployment targets, unauthenticated execution surfaces, unsupported model and performance data, and a vendored research tree whose revision and licensing basis are not established.

## Scope, method, and evidence limits

- Review date: 2026-07-21.
- Branch: `audit/aptus-legacy-forensic`.
- Scope: deployment files and manifests; auth, subscription, and API scaffolding; legal documents; the vendored `HyperTune/PyReft-Repo/` tree; model data; placeholder metrics; and generated inventory, duplicate, reference, and secret-scan evidence.
- Method for this risk track: read-only static inspection. Later bounded checks are documented in `sandbox-summary.md`; they ran parse, type, dependency-resolution, and two salvage probes. Test collection was explicitly blocked because no isolated dependency environment was installed. No container, deployment, updater, generated training artifact, model download, or training job ran.
- Baseline: 228 files, 1,879,017 bytes, 23 empty files, 38 exact-duplicate clusters containing 98 files, and 30 detected version families (`docs/audits/aptus-legacy/baseline-manifest.json:2-15`).
- Reference evidence: 143 script files were inspected statically; the generated map records 40 missing relative imports and 3 Python parse failures (`docs/audits/aptus-legacy/reference-map.json:5556-5562`).
- Secret evidence: the **high-confidence secret scan found zero findings** (`docs/audits/aptus-legacy/secret-scan.json:1-4`). This does **not** prove the absence of all sensitive information, personal data, proprietary material, low-confidence credentials, credentials in excluded formats, or externally valid identifiers.
- Confidence means confidence in the repository observation, not exploit reachability or legal conclusion:
  - **High:** directly evidenced by file content or generated inventory.
  - **Medium:** the defect is direct, but runtime reachability, external state, or upstream facts were not verified.

## Critical

### Security

#### C-SEC-01 — An intended HTTP service combines unauthenticated remote-code trust, arbitrary local paths, and unsafe upload names

- **Evidence**
  - `HyperTune/api/FastAPI/main.py:19-24` creates the application without authentication or authorization middleware.
  - `HyperTune/api/FastAPI/main.py:67-79` accepts a caller-controlled model identifier and invokes both tokenizer and configuration loading with `trust_remote_code=True`.
  - `HyperTune/api/FastAPI/main.py:133-154` accepts an arbitrary server-local dataset path.
  - `HyperTune/api/FastAPI/main.py:221-233` concatenates the unsanitized client filename into `./uploaded_datasets/{file.filename}` and writes it.
  - `HyperTune/.env.example:15-22` disables both rate limiting and API-key enforcement by default; the FastAPI file does not consume either setting.
  - `HyperTune/dockerfile_v2:17-21` intends to expose a Uvicorn service on all interfaces, although that manifest points at a nonexistent root `main` module.
- **Observed impact:** if this application is started by a corrected or alternate launcher, an unauthenticated caller can request execution of model-repository custom code, probe server-local paths, and use traversal components in an upload filename to write outside the intended directory. The current deployment files are broken, so public reachability was not established; the unsafe boundary is nevertheless explicit in the application.
- **Confidence:** High for the code paths; Medium for current external reachability.
- **Required disposition/remediation:** do not expose this application. Replace `trust_remote_code=True` with a deny-by-default, revision-pinned allowlist and isolated worker; authenticate and authorize every endpoint; constrain reads to tenant-owned storage; generate server-side upload names; enforce canonical-path containment, file type/size limits, quotas, and malware/content controls; then perform an independent security test before any deployment.

## High

### Correctness

#### H-COR-01 — No coherent TypeScript/Node or Python build graph exists

- **Evidence**
  - The generated reference map records 40 missing relative imports and 3 Python parse failures (`docs/audits/aptus-legacy/reference-map.json:5556-5562`).
  - The parse failures are `HyperTune/Complete Guide to Building & Deploying.py` at line 3 (`docs/audits/aptus-legacy/reference-map.json:3-12`), `HyperTune/src/python/core_optimizer.py` at line 1021 (`docs/audits/aptus-legacy/reference-map.json:3633-3642`), and `HyperTune/src/python/script_generator.py` at line 308 (`docs/audits/aptus-legacy/reference-map.json:4512-4521`). The broken continuation is visible at `HyperTune/src/python/core_optimizer.py:1020-1022`.
  - `HyperTune/package.json:6-9` defines only `start` and `dev`; there is no `build` or `test` script.
  - `HyperTune/.dockerfile:6-16`, `HyperTune/railway.toml:2-8`, and `HyperTune/render.yaml:3-7` nevertheless invoke `npm run build`.
  - `HyperTune/tsconfig.json:1-2` contains two hash-prefixed prose lines rather than a TypeScript JSON configuration.
  - `HyperTune/src/methods.ts:3-10` imports seven absent method modules, while `HyperTune/src/optimizer.ts:2-11` expects different exports plus two more absent modules. The missing imports are independently recorded at `docs/audits/aptus-legacy/reference-map.json:3070-3131` and `docs/audits/aptus-legacy/reference-map.json:3291-3366`.
  - `HyperTune/src/models/index.ts:58-66` imports model arrays from `llama.ts` and `gemma.ts`, but both files are empty (`docs/audits/aptus-legacy/inventory.jsonl:169-176`).
- **Observed impact:** the advertised server, optimizer, model registry, API handlers, and generated scripts cannot be built or imported as one system. Deployment failures are deterministic before business logic can be validated.
- **Confidence:** High.
- **Required disposition/remediation:** do not repair this tree incrementally. Write an Aptus-owned specification, choose one runtime and entry point, inventory only behavior supported by evidence, reimplement behind typed interfaces, and require clean static analysis and reproducible builds before porting any algorithm.

#### H-COR-02 — Auth, subscription, and premium gating are empty or interface-incompatible

- **Evidence**
  - API-key creation/verification, all three subscription endpoints, and the Prisma schema are zero-byte files (`docs/audits/aptus-legacy/inventory.jsonl:70-76` and `docs/audits/aptus-legacy/inventory.jsonl:104`).
  - `HyperTune/src/auth/index.ts` is also empty (`docs/audits/aptus-legacy/inventory.jsonl:122`).
  - `HyperTune/src/auth/api_key.ts:1-8` begins with invalid TypeScript hash prose and defines middleware taking `(req, res, next)`, but `HyperTune/api/optimize.ts:29-36` calls it as `validateApiKey(api_key)` and expects a `{valid, tier}` result.
  - `HyperTune/api/optimize.ts:4-16` imports `rateLimit`, while `HyperTune/src/auth/rate_limiting.ts:25-44` exports `rateLimiter`.
  - The intended in-memory limits are 5/day, 100/day, and 1,000,000/day (`HyperTune/src/auth/rate_limiting.ts:9-22`), which do not match the free 25/month and undefined professional limit in `HyperTune/Legal Docs/terms-of-service.md:21-31`.
- **Observed impact:** API keys cannot be created or persisted, subscriptions cannot be represented, the optimization handler cannot call the supplied auth functions correctly, and paid-tier restrictions are not enforceable.
- **Confidence:** High.
- **Required disposition/remediation:** reject the scaffold. Define the Aptus identity, entitlement, key-lifecycle, revocation, audit, and rate-limit model first; store only hashed API-key verifiers; implement it once at the transport boundary; add contract tests for every tier and failure mode; reconcile the implementation with approved commercial terms.

#### H-COR-03 — The vendored PyReft tree is incomplete as a Python package

- **Evidence**
  - `HyperTune/PyReft-Repo/pyreft/` contains no `__init__.py`; its dataset module is named `dataset.py.txt`, not `dataset.py`.
  - `HyperTune/PyReft-Repo/pyreft/mapping.py:5-8` imports `.trainer`, but the present file is `reft_trainer.py`; the missing import is recorded at `docs/audits/aptus-legacy/reference-map.json:1089-1099`.
  - `HyperTune/PyReft-Repo/pyreft/mapping.py:11-19` maps intervention names to trainer classes and references undefined `PeftModelForSequenceClassification` and `PeftModelForCausalLM`.
  - Its requirements allow broad, mutable dependency resolution (`HyperTune/PyReft-Repo/pyreft/requirements.txt:1-25`).
- **Observed impact:** the vendored directory cannot be treated as an intact, importable PyReft distribution, and its mapping behavior cannot be trusted without reconstruction against an authoritative upstream revision.
- **Confidence:** High.
- **Required disposition/remediation:** do not patch or vendor this copy into Aptus. Identify the exact authoritative upstream repository, commit/tag, package layout, release artifacts, and tests; then depend on a pinned, verified release or create a documented fork with full provenance and license notices.

### Security

#### H-SEC-01 — The Node HTTP entry point exposes resource-changing child-process operations without authentication

- **Evidence**
  - `HyperTune/server.js:10-16` enables unrestricted CORS and JSON handling but no authentication, authorization, tenant isolation, or request identity.
  - `HyperTune/server.js:18-56` dispatches caller-selected MCP function names.
  - `HyperTune/server.js:245-319` creates unbounded in-memory jobs and launches training child processes from request parameters.
  - `HyperTune/server.js:483-493` exposes `/mcp` on `0.0.0.0` through the configured port with no guard.
  - The child-process path is also wrong: every call uses `scriptPath: './python'` (`HyperTune/server.js:73-82`, `95-107`, and `272-294`), while implementations are under `HyperTune/src/python/`.
- **Observed impact:** if the Node service is launched, any network caller can attempt model registration, dataset registration, optimization, training, and export, consuming disk, network, CPU, and GPU. Current script-path drift may turn many calls into errors, but it is not a security control.
- **Confidence:** High for missing controls; Medium for a successful end-to-end operation.
- **Required disposition/remediation:** do not deploy this entry point. Put authenticated, authorized, quota-controlled job submission in front of an isolated queue; use per-tenant storage and workers; restrict model/dataset sources; bound concurrency and job-map growth; disable broad CORS; and make failures non-disclosing.

#### H-SEC-02 — Intended execution paths interpolate caller-controlled values into shell commands

- **Evidence**
  - `HyperTune/src/hypertuner/methods/base.ts:21-29`, `45-58`, `61-74` places model, dataset, output, and configuration values into `mkdir`, `python`, and `cp` command strings without shell escaping.
  - `HyperTune/src/integrations/reft_integration.ts:285-297` builds another command string from model, dataset, output, method, and JSON values.
  - `HyperTune/src/integrations/deployment/vercel.ts:108-143` interpolates task and model path values into generated JavaScript and declares mutable semver dependencies.
- **Observed impact:** these are command/code-injection sinks if their missing plumbing is restored or copied into a working service. The present TypeScript graph does not establish a currently reachable exploit path.
- **Confidence:** High for the unsafe sinks; Medium for reachability.
- **Required disposition/remediation:** do not reuse these functions. Invoke processes with argument arrays and no shell, enforce strict schemas and allowlists, use filesystem APIs instead of `mkdir`/`cp`, canonicalize all destinations, and escape generated source with structured templates or avoid source generation entirely.

### Destructive/VCS

#### H-DES-01 — The model merger fails open and can replace a database with empty or unverified data

- **Evidence**
  - `HyperTune/src/python/merge_model_data.py:27-43` converts missing or invalid scraped/manual inputs into empty dictionaries instead of aborting.
  - `HyperTune/src/python/merge_model_data.py:45-53` lets manual records silently override scraped records and then overwrites the output.
  - Its “validation” only counts fields and size buckets (`HyperTune/src/python/merge_model_data.py:56-137`); it performs no schema, source, freshness, signature, or factual validation.
  - `HyperTune/scripts/update_model_database.sh:7-34` targets nonexistent `HyperTune/Scrapers/` and `HyperTune/src/python/scrapers/` paths. Its date-only backup name (`HyperTune/scripts/update_model_database.sh:18-25`) also overwrites an earlier same-day backup.
- **Observed impact:** direct use of the merger can replace a valid output with `{}` or with manual assertions after input failure. The wrapper currently fails earlier because its scraper path does not exist, so it cannot produce the promised database.
- **Confidence:** High.
- **Required disposition/remediation:** permanently disable the updater. Any replacement must fail closed, validate inputs before touching output, preserve immutable timestamped/content-addressed backups, write atomically, require source URLs and retrieval timestamps, reject unsupported overrides, and verify the result before promotion.

#### H-DES-02 — Deployment helpers perform live external mutation without environment or confirmation safeguards

- **Evidence**
  - `HyperTune/scripts/deploy.sh:4-22` forwards an unvalidated, unquoted environment argument to four provider CLIs and invokes `vercel --prod`.
  - `HyperTune/src/integrations/deployment/vercel.ts:53-77` creates a project/deployment, and `HyperTune/src/integrations/deployment/vercel.ts:200-213` deletes a deployment by ID, with no approval boundary in the provider interface.
  - `HyperTune/DeploymentOptions/AWS/config.yaml:2-18` binds an Elastic Beanstalk environment to `main` and Git source control despite the legacy folder being intentionally ignored by Aptus.
- **Observed impact:** a mistaken invocation can mutate or delete external infrastructure with whatever account the local CLI/API token controls. There is no dry run, target summary, branch/commit binding, protected-environment check, or human confirmation.
- **Confidence:** High for mutating behavior; Medium for credential availability.
- **Required disposition/remediation:** do not run these helpers. Rebuild deployment through reviewed IaC and CI with immutable artifacts, environment protections, least-privilege credentials, plan/apply separation, explicit production approval, audit logs, and deletion protection.

### Dependency/supply-chain

#### H-SUP-01 — Dependency resolution is unpinned, split across incompatible manifests, and missing declared runtime packages

- **Evidence**
  - No npm, Yarn, pnpm, Poetry, Pipenv, or uv lockfile exists anywhere in `HyperTune/`.
  - `HyperTune/package.json:10-22` uses caret ranges and does not declare packages imported by the TypeScript sources, including `@modelcontextprotocol/sdk`, `next`, `@prisma/client`, `rate-limiter-flexible`, and `axios`.
  - `HyperTune/scripts/setup.sh:1-3` installs current npm packages without versions and mutates the local dependency state.
  - `HyperTune/requirements.txt:3-32` uses only lower bounds for application, ML, testing, documentation, and deployment packages.
  - The alternate sets conflict in purpose and version floor: `HyperTune/requirements_v2.txt:1-12`, `HyperTune/src/python/requirements.txt:1-15`, and `HyperTune/PyReft-Repo/pyreft/requirements.txt:1-25`.
  - Container bases are tags rather than immutable digests (`HyperTune/dockerfile:1-23`, `HyperTune/dockerfile_v2:1-12`), and one build executes a remote NodeSource setup script (`HyperTune/dockerfile:12-15`).
- **Observed impact:** two installs can resolve different transitive code; declared manifests cannot satisfy the checked-in imports; and container contents can drift without source changes. No SBOM, integrity lock, or reproducible build evidence is present.
- **Confidence:** High.
- **Required disposition/remediation:** create a new minimal dependency set from the chosen Aptus implementation, lock exact direct and transitive versions with hashes, pin container digests, generate an SBOM, run license/vulnerability policy checks, and update through reviewed automation. Do not derive a production lock by executing these legacy installers.

### Licensing/provenance

#### H-LIC-01 — The repository does not establish a distributable license or the provenance of vendored research code

- **Evidence**
  - `HyperTune/LICENSE:1-2` contains only `# LICENSE` and `# MIT`; it is not the MIT license text and names no copyright holder.
  - No `LICENSE`, `COPYING`, `NOTICE`, authors file, package metadata, submodule metadata, commit identifier, or source manifest exists inside `HyperTune/PyReft-Repo/`.
  - `HyperTune/PyReft-Repo/loreft/README.md:1-5` says the example is based on a Stanford PyReft script.
  - The same README says “We copy everything” from LLM-Adapters and identifies additional datasets (`HyperTune/PyReft-Repo/loreft/README.md:9-31`).
  - `HyperTune/Legal Docs/eula.md:33-39` asserts ownership and proprietary trade secrets across the software, while the audited tree contains identified third-party-derived material.
- **Observed impact:** the files do not establish who may copy, modify, distribute, host, or commercially use the legacy code or each vendored component. The EULA language cannot supply missing third-party permissions.
- **Confidence:** High that provenance/license evidence is missing; no conclusion is made here about the actual upstream licenses.
- **Required disposition/remediation:** block reuse and distribution pending counsel and provenance review. Treat every upstream-license assertion as requiring verification against authoritative upstream repositories/releases and the exact incorporated revision. Record source URLs, commit hashes, file-level origins, applicable license texts, notices, modifications, dataset terms, and redistribution obligations before any code is considered.

#### H-LIC-02 — The legal documents are unpublishable templates and make unsupported operational commitments

- **Evidence**
  - The EULA leaves company, state, address, email, and date placeholders (`HyperTune/Legal Docs/eula.md:5`, `HyperTune/Legal Docs/eula.md:71`, and `HyperTune/Legal Docs/eula.md:81-85`) while promising a 90-day limited warranty (`HyperTune/Legal Docs/eula.md:53-57`) and separately disclaiming other warranties (`HyperTune/Legal Docs/eula.md:59`).
  - The terms leave effective date, service description, professional limit, company, jurisdiction, and contact fields blank (`HyperTune/Legal Docs/terms-of-service.md:5-13`, `HyperTune/Legal Docs/terms-of-service.md:23-31`, and `HyperTune/Legal Docs/terms-of-service.md:113-128`).
  - The privacy policy claims account, billing, password, analytics, advertising, cookie, firewall, retention, and data-rights practices (`HyperTune/Legal Docs/privacy-policy.md:15-43`, `HyperTune/Legal Docs/privacy-policy.md:95-109`, and `HyperTune/Legal Docs/privacy-policy.md:111-173`) while providing only placeholder contact channels (`HyperTune/Legal Docs/privacy-policy.md:154-161`).
  - Exact duplicate copies of all three legal documents exist (`docs/audits/aptus-legacy/duplicate-clusters.json:14-20`, `docs/audits/aptus-legacy/duplicate-clusters.json:159-165`, and `docs/audits/aptus-legacy/duplicate-clusters.json:213-219`).
- **Observed impact:** there is no identifiable contracting party, valid contact route, settled jurisdiction, complete pricing limit, or repository evidence for many stated data practices. Publishing these files would create contradictory and potentially misleading commitments.
- **Confidence:** High for incompleteness and lack of support within the audited scope.
- **Required disposition/remediation:** do not publish or adapt these templates. Have qualified counsel draft Aptus-specific terms from the actual product, data map, subprocessors, security controls, retention schedule, support/warranty policy, billing system, and verified third-party licenses; keep one approved canonical version with effective-date/version control.

### Fabricated/unsupported claims

#### H-CLAIM-01 — Training and comparison paths return fixed or heuristic values as if they were measured results

- **Evidence**
  - `HyperTune/src/python/tune_service.py:9-17` states that it only defines an interface, yet LoRA returns `success: True`, loss `0.1/0.2` (`HyperTune/src/python/tune_service.py:22-45`); DoRA adds fixed efficiency `0.85` (`HyperTune/src/python/tune_service.py:47-72`); QLoRA adds fixed `memory_saved: "75%"` (`HyperTune/src/python/tune_service.py:74-100`); and AdaLoRA/ReFT repeat fixed success/loss values (`HyperTune/src/python/tune_service.py:102-154`).
  - `HyperTune/src/hypertuner/task-configs.ts:413-463` exposes method comparisons whose four metrics are constant `0.95`, `0.8`, `0.9`, and `0.85` for every method/model/task.
  - `HyperTune/src/hypertuner/methodSelector.ts:22-111` presents fixed “accuracy impact” values and dimensional heuristics as memory/training estimates without calibration data.
  - `HyperTune/api/comparison.ts:72-85` computes “performance” from rank and a hard-coded `0.9` multiplier, capped at `0.99`.
  - `HyperTune/src/integrations/training/runpod.ts:130-143` reports progress from an assumed one-hour duration rather than provider training state.
- **Observed impact:** callers can receive synthetic success, loss, memory savings, accuracy, speed, and progress values that are indistinguishable from measured outcomes in the response shape. These values cannot support product, scientific, customer, pricing, or safety decisions.
- **Confidence:** High.
- **Required disposition/remediation:** remove every synthetic metric from user-facing paths. Mark prototypes incapable of producing results, define metric provenance in schemas, return measured values only with run IDs and artifacts, publish calibration/error bounds for estimators, and validate against held-out benchmark runs.

#### H-CLAIM-02 — “Optimal,” benchmark, and production-readiness claims are unsupported by the inspected evidence

- **Evidence**
  - The legacy guide says the system is production-ready and “eliminates guesswork” (`HyperTune/hypertune-mcp-guide.md:1-16`) and concludes that it is production-ready and saves “countless hours” (`HyperTune/hypertune-mcp-guide.md:2693`).
  - `HyperTune/src/optimizer.ts:256-265` describes outputs as optimal and “based on benchmarks,” but no benchmark dataset, protocol, results artifact, statistical analysis, or calibration record is present.
  - `HyperTune/src/llm_integrations.ts:18-72` labels hard-coded ranks/layers “optimal”; model-structure inference is explicitly a placeholder (`HyperTune/src/llm_integrations.ts:205-210`), and budget-to-rank conversion is explicitly a simple heuristic (`HyperTune/src/llm_integrations.ts:261-265`).
  - `HyperTune/deploy/Summary of Implementation.md.txt:4-26` claims model detection, optimal selection, multiple advanced methods, and state-of-the-art backend support despite missing modules and placeholder implementations.
- **Observed impact:** documentation materially overstates completeness, validation, and expected outcomes. The claims are contradicted by static build failures and synthetic metrics.
- **Confidence:** High that the claims are unsupported by this repository; no claim is made that no external evidence exists.
- **Required disposition/remediation:** withdraw all performance, optimality, benchmark, savings, security, and production-readiness language unless each claim has an owner, precise definition, versioned methodology, reproducible artifact, uncertainty statement, and legal/product approval.

#### H-CLAIM-03 — Model facts are unattributed manual assertions and override scraped data without factual validation

- **Evidence**
  - `HyperTune/HyperTune-NEW_stuff_05-16-25/manual-models.json:2-22` assigns 1,700B and 1,800B parameter counts to OpenAI models without citing an authoritative parameter source; `HyperTune/HyperTune-NEW_stuff_05-16-25/manual-models.json:24-66` does the same for 1,000B, 250B, 80B, and 500B Anthropic/Google counts. Every record is labeled only `source: "manual"`.
  - The scraper infers parameter counts from names when metadata is absent (`HyperTune/HyperTune-NEW_stuff_05-16-25/model-scraper.py:180-230`) and lets manual models replace existing records (`HyperTune/HyperTune-NEW_stuff_05-16-25/model-scraper.py:540-554`).
  - Normalization merely fills absent fields with `None` or `unknown` (`HyperTune/HyperTune-NEW_stuff_05-16-25/model-scraper.py:599-637`).
  - The checked-in manual filename/location does not match the expected `HyperTune/Scrapers/data/manual_models.json`, and no `HyperTune/Scrapers/` directory exists.
- **Observed impact:** parameter counts, descriptions, authorship labels, and release facts can enter recommendation logic without authoritative sources, confidence, retrieval date, model revision, or license evidence; the current updater cannot find the checked-in manual data anyway.
- **Confidence:** High that the assertions are unsupported in this repository. Their factual truth was not externally verified.
- **Required disposition/remediation:** quarantine these records. Require authoritative first-party/model-card sources, immutable URLs or snapshots, retrieval dates, model/version identifiers, field-level provenance, confidence, review status, and schema validation. Do not infer undisclosed closed-model parameter counts or present them as facts.

### Deployment drift

#### H-DRIFT-01 — Deployment files select different applications, ports, paths, resources, and nonexistent health/build targets

- **Evidence**
  - `HyperTune/.dockerfile:1-16` builds Node and exposes 8080, but invokes the absent `npm run build`.
  - `HyperTune/dockerfile:1-32` installs both ecosystems and starts the unauthenticated Node server on 8080.
  - `HyperTune/dockerfile_v2:1-21` exposes 8000 and starts `uvicorn main:app`, but no root `main.py` exists.
  - `HyperTune/deploy/dockerfile.txt:1-38` is Markdown, exposes 8787, and also invokes the absent build script.
  - `HyperTune/fly.toml:1-22` routes 8080 with a 16 GB VM, while `HyperTune/fly_v2.toml:1-29` routes 8000, requests `Dockerfile`, and defines no equivalent VM resources.
  - `HyperTune/railway.toml:2-9` calls the absent build script and probes `/health`; `HyperTune/server.js:483-493` defines only `/mcp`.
  - `HyperTune/DeploymentOptions/kubernotes/deployment.yaml:16-34` uses the placeholder image `gcr.io/PROJECT_ID/...`, port 8080, and a 512 MiB memory limit for an ML service.
  - Exact duplicate deployment files exist in multiple locations (`docs/audits/aptus-legacy/duplicate-clusters.json:23-38`, `docs/audits/aptus-legacy/duplicate-clusters.json:186-192`, `docs/audits/aptus-legacy/duplicate-clusters.json:222-228`, and `docs/audits/aptus-legacy/duplicate-clusters.json:276-282`).
- **Observed impact:** providers may build different runtimes, fail before startup, route to the wrong port, fail health checks, or deploy an unauthenticated executor. There is no canonical artifact or deployment contract.
- **Confidence:** High.
- **Required disposition/remediation:** delete these as deployment candidates. Define one Aptus service contract, produce one immutable image from a locked build, add authenticated readiness/liveness endpoints, set measured resource limits, scan the image, and deploy only through reviewed IaC.

### Test evidence

#### H-TEST-01 — The repository contains test-shaped files but no credible release evidence

- **Evidence**
  - `HyperTune/package.json:6-9` has no test command.
  - `HyperTune/src/hypertuner/test/test_hypertune_mcp.ts:5-50` is a logging script with no assertions; its catch block only logs errors and does not set a failing exit status (`HyperTune/src/hypertuner/test/test_hypertune_mcp.ts:51-56`).
  - Python “integration” tests substitute `MockScriptGenerator` for the real generator (`HyperTune/tests/integration/test_workflow.py:19-56`) and mock GPU behavior (`HyperTune/tests/integration/test_workflow.py:58-72`).
  - The Python optimizer and generator fail static parsing (`docs/audits/aptus-legacy/reference-map.json:3633-3642` and `docs/audits/aptus-legacy/reference-map.json:4512-4521`), so the workflow tests do not establish those production paths.
  - The unit and integration suites are duplicated byte-for-byte, including `tests/test_optimizer.py`, `tests/run_tests.py`, and the full `tests/integration/` set (`docs/audits/aptus-legacy/duplicate-clusters.json:5-12`, `docs/audits/aptus-legacy/duplicate-clusters.json:96-102`, `docs/audits/aptus-legacy/duplicate-clusters.json:249-273`, `docs/audits/aptus-legacy/duplicate-clusters.json:285-300`, and `docs/audits/aptus-legacy/duplicate-clusters.json:333-339`).
  - The generated reference tool did not parse-check JavaScript/TypeScript syntax; for example, it marks those files `parse_status: "not_checked"` (`docs/audits/aptus-legacy/reference-map.json:1371-1383`).
- **Observed impact:** there is no repository evidence that any deployable entry point, auth flow, subscription rule, model updater, container, provider integration, legal promise, or security control works. Duplicate and mocked tests inflate apparent volume without broadening coverage.
- **Confidence:** High.
- **Required disposition/remediation:** treat test status as **unknown/not run**, not passing. For a replacement, require hermetic unit tests, API/auth contract tests, fixture provenance, property tests for estimators, security regression tests, container smoke tests, deployment checks, and benchmark/calibration artifacts in CI.

## Medium

### Destructive/VCS

#### M-VCS-01 — The legacy source is intentionally outside Aptus version control

- **Evidence**
  - Aptus ignores the entire legacy folder as an external audit input (`.gitignore:1-2`).
  - The generated baseline supplies a point-in-time manifest hash but no upstream commit or archive identity (`docs/audits/aptus-legacy/baseline-manifest.json:5-15`).
- **Observed impact:** Git history cannot establish when, why, or by whom a legacy file changed, and ordinary status/diff review will not detect later edits inside the ignored folder. The manifest is useful integrity evidence only for the captured snapshot.
- **Confidence:** High.
- **Required disposition/remediation:** keep the folder quarantined and ignored. Preserve the input as a read-only, content-addressed archive with acquisition source, timestamp, owner, checksums, chain of custody, and access policy; compare future inspections against that archive rather than silently editing this copy.

### Correctness

#### M-COR-01 — Duplicate and version-family sprawl makes file selection ambiguous

- **Evidence**
  - The baseline records 38 duplicate clusters, 98 duplicate files, and 30 version families (`docs/audits/aptus-legacy/baseline-manifest.json:2-15`).
  - Three differently named “complete guide” files are byte-identical (`docs/audits/aptus-legacy/duplicate-clusters.json:77-85`).
  - Node server, Docker, Fly, Railway, Render, requirements, CLI, Python service, tests, legal documents, and ReFT guides all have duplicate or numbered copies throughout `docs/audits/aptus-legacy/duplicate-clusters.json:5-367`.
- **Observed impact:** a maintainer can fix one copy while launchers, documentation, or deployment tooling consume another; names such as ` 2`, `_v2`, `.txt`, and hidden `.dockerfile` do not establish authority or chronology.
- **Confidence:** High.
- **Required disposition/remediation:** do not deduplicate in place. During reimplementation, create a decision record for each capability, select behavior by evidence rather than filename, preserve rejected variants in the forensic archive, and keep exactly one owned source for each Aptus contract.

### Deployment drift

#### M-DRIFT-01 — Instructions contain personal absolute paths and nonexistent project layout

- **Evidence**
  - `HyperTune/fly-deploy.txt:17-27` instructs users to create and enter `/Users/biscuit/Desktop/tunerepo/HyperTune`.
  - `HyperTune/README.md:24-63` documents one layout, while `HyperTune/server.js:73-82` expects a root `python/` directory and updater scripts expect absent `Scrapers/` and `src/python/scrapers/` directories.
  - `HyperTune/HyperTune-NEW_stuff_05-16-25/dockerfile.txt:10-17` copies a nonexistent `Scrapers/data/base_models.json`.
- **Observed impact:** instructions are not portable and cannot reproduce the filesystem expected by the launchers or containers.
- **Confidence:** High.
- **Required disposition/remediation:** do not reuse these instructions. Generate Aptus deployment documentation from tested CI commands and the canonical repository layout, with no developer-specific paths or undeclared generated inputs.

## Required release gates for any salvage

No legacy component should enter Aptus merely because a local syntax fix makes it run. Salvage requires all of the following:

1. **Provenance gate:** exact source/revision, author, modification history, authoritative license, notices, and dataset/model terms.
2. **Correctness gate:** one specification and entry point; no missing imports, empty required modules, parser errors, synthetic outputs, or ambiguous duplicate authority.
3. **Security gate:** threat model; no untrusted remote code; no shell interpolation; authenticated/authorized APIs; tenant isolation; bounded jobs/uploads; secrets and key lifecycle; independent review.
4. **Supply-chain gate:** locked dependencies and hashes, pinned image digests, SBOM, vulnerability/license policy, reproducible build.
5. **Evidence gate:** measured and traceable metrics, documented estimator uncertainty, reproducible benchmarks, and CI artifacts.
6. **Deployment gate:** one immutable artifact and reviewed IaC with environment protection, health checks, rollback, audit logs, and deletion safeguards.
7. **Legal gate:** counsel-approved Aptus terms/privacy/EULA derived from actual behavior and verified third-party rights.

Until those gates are met, the legacy folder is forensic evidence, not Aptus product code.
