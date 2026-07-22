# Reference and TO-REVIEW reconciliation

Status: reviewed against the Aptus v0.2 source tree, salvaged, and removed from
the working tree on 2026-07-22.

This ledger records every file supplied in `Reference/` and the former
`TO-REVIEW/` staging folder. The `Reference/` research packet remains in the
repository. The `TO-REVIEW/` files were removed after review because none was
safe to import as executable code. Their exact digests and dispositions remain
below, and Git commit `b1314b3` retains the last tracked snapshot.

The disposition terms mean:

- **Integrate**: use the concept now, subject to the named evidence and runtime
  contracts.
- **Roadmap**: preserve the requirement, but do not present it as a current
  capability.
- **Archive**: retain the historical requirement in this ledger, not a raw
  staging copy in the current tree.
- **Reject**: do not import the factual claim or code into an executable path.
  Rejected staging files are removed after their disposition is recorded.

## Review rules

1. Runtime behavior is authoritative only when it appears in the typed domain,
   planner, compiler, validator, and tests. A README is not proof.
2. A paper establishes a method, not compatibility with Aptus's pinned stack,
   a particular model, or a particular GPU.
3. A provider or library capability is rechecked against the pinned version
   before it enters an executable strategy.
4. Qualitative words such as "optimal," "perfect," and "guaranteed" are not
   accepted without an objective, comparison set, measurement protocol, and
   evidence record.
5. Generated or imported code does not execute until its inputs, paths,
   process boundary, resource lease, cancellation behavior, and artifact
   contract are validated.

The current runtime authorities are
[`src/aptus/domain.py`](../../src/aptus/domain.py),
[`src/aptus/planning.py`](../../src/aptus/planning.py), and
[`src/aptus/generation.py`](../../src/aptus/generation.py). The normalized method
model derived from this review is in
[`docs/methodology/method-taxonomy.md`](../methodology/method-taxonomy.md).

## Reference folder

### `Reference/top-50-llm-training-methods.pplx.md`

**Disposition: Integrate the taxonomy and research index. Revalidate each
library mapping before implementation.**

Useful material:

- It separates objectives, parameterizations, recipes, pipelines, modifiers,
  full-parameter update strategies, and infrastructure.
- It correctly treats LoRA as a parameterization and QLoRA as a composed
  recipe. The QLoRA paper defines the recipe as gradients through a frozen,
  four-bit base into LoRA adapters, with NF4, double quantization, and paged
  optimizers as key memory techniques
  ([Dettmers et al.](https://arxiv.org/abs/2305.14314)).
- Its compiler insertion points are useful: model loading, parameter
  selection, dataset contract, objective, optimizer, distribution,
  evaluation, and export.
- Its decision tree is a good research intake aid. It is not a planner policy
  until every branch has facts, compatibility rules, estimates, and a pilot
  contract.

Limits and corrections:

- The rank order is editorial. It is not model-quality evidence and must not
  become an Aptus score.
- "Native" library support is version-sensitive. The file describes PEFT
  v0.17.0 plus a then-current main branch, while Aptus pins its own dependency
  set. Every API mapping needs a pinned-runtime test.
- The generic Python is pseudocode. It does not bind model revision, dataset
  digest, environment, memory admission, checkpoint continuation, export
  integrity, or distributed failure behavior.
- A paper result such as a 65B run on one 48 GB GPU describes the paper's
  configuration. It is not a universal capacity rule.
- The list documents 50 research items. Aptus does **not** implement or claim
  support for 50 methods.

### `Reference/FineTuneX.README.md`

**Disposition: Archive the product history. Move the interface decomposition
to the roadmap.**

Useful material:

- Separate planning and artifact generation from compatibility inspection.
- Preserve a future MCP or provider adapter over the same typed Aptus service
  contracts.
- Preserve the workflow from explicit facts to scripts, compatibility checks,
  and user-controlled execution.

Limits and safety issues:

- The two MCP services, pricing, support channels, hosted documentation, and
  deployment options are not implemented by the current repository.
- "Optimal," "perfectly configured," and "guaranteed compatibility" exceed
  the available evidence.
- The examples use old, unbound package versions and do not pin a model commit
  or dataset digest.
- The compatibility service can suggest environment changes without an
  approval, integrity, or rollback contract.

### `Reference/Fine-Tuning_Methods.md`

**Disposition: Reject as factual authority. Retain only as an unverified name
intake list.**

Confirmed factual errors include:

- It expands `FFT` as "Fast Fine-Tuning." In this project, `full` means
  full-parameter fine-tuning. `FFT` is too ambiguous to use as a contract term.
- It expands ReFT as "Recursive Fine-Tuning." The primary paper defines ReFT
  as **Representation Finetuning** over interventions on hidden
  representations ([Wu et al.](https://arxiv.org/abs/2404.03592)).
- It describes LoReFT as low-rank regularization. LoReFT is Low-rank Linear
  Subspace ReFT, a representation intervention in the ReFT family.
- It expands DoRA as "Dynamic Optimization for Regularization Adaptation."
  The primary paper defines DoRA as **Weight-Decomposed Low-Rank Adaptation**,
  which separates magnitude and direction and applies LoRA to directional
  updates ([Liu et al.](https://arxiv.org/abs/2402.09353)).
- It calls QLoRA "Quantization-based Fine-Tuning" and omits the frozen NF4
  base, LoRA adapters, double quantization, and paged-optimizer composition.

The remaining entries are mostly uncited, underspecified, duplicated, or
incomplete. Names such as DoReFT, FishDip, FAR, CIAT, KODA, MerA, PHA, and PaFi
must not enter the catalog until a primary source, exact mechanism, maintained
implementation, and distinctness test are recorded.

### `Reference/hparam_methods_reference.md`

**Disposition: Roadmap research notes. Reject the numeric values as planner
defaults.**

Useful material:

- It identifies configuration dimensions worth modeling, including LoRA rank,
  alpha, dropout, optimizer, warmup, and SAM perturbation scale.
- It suggests future experiment families for AdaLoRA, BiLoRA, ShareLoRA,
  BitFit, Intrinsic SAID, and SAM.

Limits and corrections:

- The hyperparameter ranges and formulas have no citations, model families,
  dataset conditions, objective, or evaluation protocol.
- "PagedAdamW8bit" and the 65B memory statement must not replace Aptus's
  explicit optimizer and measured pilot contracts.
- The AdaLoRA proportionality and the Intrinsic SAID description are
  compressed heuristics, not implementation specifications.
- The file cannot support a claim of hyperparameter optimization. At most, it
  supplies priors that must be labeled, tested, and compared.

## Former TO-REVIEW folder

Every path in this section is historical. The staging folder no longer exists
in the current tree. The product requirements that survived review now live in
the typed method registry, planner, compiler, execution service, UI, and the
roadmap named below.

### `TO-REVIEW/SystemArchitecture.md`

**Disposition: Roadmap architecture sketch.**

Useful material:

- Preserve explicit seams for a method registry, fact analyzer, trainer,
  evaluator, exporter, benchmark definitions, and hardware manager.
- Keep method metadata separate from implementation modules.

Limits:

- Most listed files do not exist. The sketch does not define schemas,
  provenance, compatibility versions, state transitions, leases, or artifact
  evidence.
- Its folders mix method axes. Quantization and mixture-of-experts are not
  peers of objectives, parameter scopes, and distribution strategies.
- "All 25 methods" is an aspiration, not an implementation count.

### `TO-REVIEW/core/registry.ts`

**Disposition: Roadmap the registry concept. Reject this implementation.**

Useful material:

- A discoverable, filterable registry with paper references, use cases, and
  default configuration is a sound interface goal.

Limits and safety issues:

- The Markdown heading makes the file invalid TypeScript.
- `FineTuningMethod` and three imported implementations are missing. Only four
  registrations are sketched, despite the claim of 25.
- There is no schema version, evidence revision, implementation status,
  compatibility predicate, duplicate-name check, or compiler binding.
- `memoryUsage` is a string in the DoRA sketch, but the registry passes it to
  `parseInt`, so memory filtering cannot be trusted.

### `TO-REVIEW/core/analyzer.ts`

**Disposition: Integrate the explainable filter-and-rank shape. Reject the
scoring algorithm.**

Useful material:

- Profile the dataset before recommending a method.
- Filter candidates by hard constraints, rank survivors, and return rationale.

Limits and safety issues:

- Method inclusion and exclusion are hard-coded strings. Scores of five and
  three have no evidence or units.
- It does not model model revision, real VRAM, host RAM, disk, quantization
  capability, exact batch arithmetic, precision, or distribution.
- It can dereference `recommendedMethods[0]` when no candidate survives.
- The prose rationale says "best balance" without exposing a comparison basis
  or uncertainty.

### `TO-REVIEW/core/trainer.ts`

**Disposition: Integrate lifecycle requirements. Reject this implementation.**

Useful material:

- Check resources before allocation, persist run inputs, use a unique run
  directory, and release resources in `finally`.

Limits and safety issues:

- The Markdown heading makes the file invalid TypeScript. `GPUManager` and the
  method implementations are missing.
- It mutates process-global `CUDA_VISIBLE_DEVICES` inside a request handler.
  Concurrent jobs could affect each other.
- A caller controls `output_dir`. There is no path containment, symlink,
  ownership, quota, or no-clobber policy.
- Timestamp IDs can collide. Parameter writes are not atomic and may expose
  secrets.
- The method's returned success and artifact paths are trusted without
  aggregate process exit, manifest, hash, or structural verification.
- There is no host-global GPU lease, process-group cancellation, restart
  reconciliation, or immutable run evidence.

### `TO-REVIEW/methods/low_rank/dora.ts`

**Disposition: Reject the executable code. Integrate only the primary paper
identity and interface questions.**

Useful material:

- The name, paper URL, target-module input, resource-estimate seam, and
  structured result shape are useful catalog fields.

Limits and safety issues:

- The Markdown heading makes the file invalid TypeScript. The interface and
  referenced Python script are missing.
- It builds a shell command from caller-controlled model, dataset, output, and
  target-module values, then runs `execSync`. Quoting is incomplete, so this is
  a command-injection path. Aptus must use an argument vector and a controlled
  interpreter.
- `regularization_strength` does not describe DoRA's magnitude-and-direction
  mechanism and is not justified by the cited paper.
- Hard-coded model sizes, the 20 percent memory multiplier, batch-size memory
  term, CPU count, defaults, and fallback model size are unsupported.
- It has no immutable model revision, dataset hash, environment binding,
  admission refresh, distributed contract, cancellation, checkpoint proof, or
  export attestation.

### `TO-REVIEW/server.ts`

**Disposition: Roadmap an adapter layer. Reject as a deployable server.**

Useful material:

- The proposed surface identifies future service capabilities: analyze,
  inspect method metadata, train, evaluate, export, and deploy.

Limits and safety issues:

- The file is invalid TypeScript and imports missing components. It is not a
  runnable MCP server in this repository.
- Request schemas omit immutable model revision, license permission,
  provenance, dataset digest, host facts, run identity, and approval tokens.
- Training and deployment are high-impact actions but have no authentication,
  authorization, path containment, idempotency, GPU lease, job polling,
  cancellation, or audit record.
- Evaluation, GGUF/ONNX export, and provider deployment are declared without
  implementations or validation contracts.
- One endpoint per method duplicates the generic method contract and can drift.

### `TO-REVIEW/Client_integration_example.ts`

**Disposition: Archive the intended sequence. Roadmap an asynchronous,
evidence-bound client.**

Useful material:

- The high-level sequence of analyze, approve/train, evaluate, and export is a
  useful future user journey.

Limits and safety issues:

- It selects the first recommendation and starts training without presenting
  alternatives, assumptions, pilot state, or explicit approval.
- It assumes synchronous training. It has no job identity, polling,
  cancellation, recovery, or terminal artifact verification.
- It trusts returned paths and immediately evaluates and converts them.
- It provides no authentication or transport-security contract.

### `TO-REVIEW/README-v2.md`

**Disposition: Archive the historical positioning. Integrate selected product
requirements into the roadmap.**

Useful material:

- Keep local artifact generation, optional local execution, and future cloud
  provider connectors as separate choices.
- Preserve the goal of reducing method selection and configuration work.

Limits and corrections:

- It says FineTuneX does not run training, while current Aptus includes a
  guarded local orchestration path.
- The 25-method list, cloud connectors, pricing, hosted documentation, and
  one-click deployment claims are not present capabilities.
- The examples omit immutable revisions, evidence, real admission, and pilot
  authorization.
- A `latest` container tag is not reproducible.

### `TO-REVIEW/v3_README.md`

**Disposition: Archive as a duplicate historical draft.**

It repeats `Reference/FineTuneX.README.md`, including an additional duplicated
opening section and placeholder image text. The useful service decomposition
and future adapter concept are already captured above. The same unsupported
claims about 25 methods, perfect configuration, guaranteed compatibility,
pricing, documentation, and deployment apply.

### `TO-REVIEW/docker-compose.yml`

**Disposition: Reject as deployment configuration. Roadmap container
isolation only after a security and GPU scheduling design.**

Limits and safety issues:

- The referenced Dockerfile and `prometheus.yml` are missing.
- Images are unpinned. Host ports expose the service, Grafana, and Prometheus.
- Writable model and dataset mounts have no ownership or isolation policy.
- A hard-coded GPU and Compose reservation do not implement the Aptus
  host-global lease or prove exclusive access.
- There are no health checks, secrets, authentication, resource ceilings,
  worker separation, or artifact-retention rules.

### `TO-REVIEW/fly.toml`

**Disposition: Reject as deployment configuration.**

The file names a public HTTP service and GPU shape but supplies no image/build,
persistent storage, secrets, health check, authentication, worker boundary,
model cache policy, or evidence that the declared GPU configuration is valid
for the provider. It must not be advertised as a working deployment.

### `TO-REVIEW/railway.json`

**Disposition: Reject as deployment configuration.**

The first `#` line makes the file invalid JSON. The repository has no Node
package to satisfy its `npm` commands. The GPU and volume schema is unverified,
and there are no health, secret, authentication, storage-integrity, or job
isolation contracts.

### `TO-REVIEW/.DS_Store`

**Disposition: Reject as repository content and exclude from commits during
repository hygiene.**

This was a 6,148-byte Finder metadata file. It contained no product, research,
or implementation material. It was reviewed as binary metadata, recorded in
the digest inventory below, and deleted with the rest of the staging folder.
The repository-wide `.gitignore` excludes future copies.

## Accepted product requirements

The review yields the following requirements without importing unsafe code:

1. The versioned registry in `src/aptus/methods/` separates research identity
   from executable status and supplies the API workbench catalog.
2. Candidate selection must be a fact-driven comparison, not a hard-coded top
   method or a claim of universal optimality.
3. Compatibility inspection must bind the exact model commit, dataset digest,
   hardware observation, dependency environment, and generated bundle.
4. Evaluation and export need independent typed contracts. A finite pilot loss
   is operational evidence, not target-quality proof.
5. Provider and MCP support should be thin adapters over the same service and
   job contracts. They must not bypass authorization, leases, or verification.
6. A method enters the executable catalog only with a compiler implementation,
   pinned-runtime tests, conservative estimates, measured preflight, bounded
   pilot, artifact verifier, and explicit unsupported cases.

## Salvage implementation

The folder review produced concrete changes instead of leaving historical
drafts in place:

- `src/aptus/methods/contracts.py` defines a versioned method descriptor whose
  lifecycle is independent of whether the planner may select it.
- `src/aptus/methods/registry.py` exposes the four guarded executable methods
  and separately labels DoRA, BitFit, AdaLoRA, and ShareLoRA experimental, plus
  LoReFT, AFLoRA, and BiLoRA research-only. All seven are nonselectable and have
  an explicit blocker and required proof.
- `/api/v1/bootstrap` returns the same registry to the workbench. The method
  preference control is populated from selectable descriptors, while a
  readiness board explains why the other methods remain unavailable.
- The generated trainer rejects an empty, non-finite, or method-scope-invalid
  trainable set and binds a name-shape-dtype census to measured evidence.
- Full-run dataset splitting respects an explicit `split_group` so related
  corpus chunks cannot cross train and evaluation partitions. It also binds
  canonical and assignment digests, reports realized split error, and rejects
  mutation or distributed disagreement.
- Provider, evaluator, exporter, cloud, and MCP ideas remain typed roadmap
  seams. The unauthenticated server and deployment sketches were not retained.

## Complete reviewed snapshot

These SHA-256 digests bind this ledger to the exact reviewed files, including
the removed staging snapshot.

| Path | SHA-256 |
|---|---|
| `Reference/Fine-Tuning_Methods.md` | `124aa4de9dd725dc4d352342842712a9ad18bc137a3bd404ea67bb08135a28fa` |
| `Reference/FineTuneX.README.md` | `08ae442919e23b331f76653cd02f07abef66638411e1bd5f413f59797781adff` |
| `Reference/hparam_methods_reference.md` | `d28402785dec4091ea7ea31e6c1cb9393219b6400440e275cf5f8975fd42f633` |
| `Reference/top-50-llm-training-methods.pplx.md` | `4b834700918ee6f2d5eb37560a4f484d266eea2566bd1ba39c5bb2e7123315bc` |
| `TO-REVIEW/.DS_Store` | `646f153463cf1510143b7a48a5e815725bf729349b636c2ded0e63b448f70d42` |
| `TO-REVIEW/Client_integration_example.ts` | `71ce0a7886b8b89ae922b81fde5b9b68acbcb02703a8c43448b559098203965e` |
| `TO-REVIEW/README-v2.md` | `4829429c6bd678383585ebd17ba861b1e8169140dd3fd8efcf67af6634ed89ad` |
| `TO-REVIEW/SystemArchitecture.md` | `e31810cac65ef8c1ac58a66b2b6ac58161cd9421e6c20f18e9cc7bd0601e6930` |
| `TO-REVIEW/core/analyzer.ts` | `bd23db06c5c5216cb7d8c6e315a561ad28949b9ee8126f79c32368ed523d1243` |
| `TO-REVIEW/core/registry.ts` | `416c390c0a3092422384c6093dec24e1c92bea15d813ecae77e9c7307970ecd8` |
| `TO-REVIEW/core/trainer.ts` | `1b7da93d602586703770cf0f1499a2a5f4e9bf13facd70ac189d99b933e283f0` |
| `TO-REVIEW/docker-compose.yml` | `c592a1a609ec8e0bb791d2d6e6699e656549e3224c707d0a78ce612cae95ed9d` |
| `TO-REVIEW/fly.toml` | `2396c0f5e58801c588b8a9419a07dbf55443630592889946f9d6037afd6d5718` |
| `TO-REVIEW/methods/low_rank/dora.ts` | `333276a29872c0cace331986b46c7b12a00b19442c3fc0632cb6c9d99e162bd7` |
| `TO-REVIEW/railway.json` | `33994dec931fcad23b82c35c7787a9cd31a30919fbe6512d73df20c6c8d48948` |
| `TO-REVIEW/server.ts` | `2e66b15867845447ffad1484c8197265fee10a386678e4d723d0f3338b53ec42` |
| `TO-REVIEW/v3_README.md` | `84e4fb92eab731c3a40c60fde29c5c63ce0cde49895b1eac09f5c988f607167a` |
