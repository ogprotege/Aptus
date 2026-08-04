# Release Evidence Template

> **Status:** Active template | **Audience:** Maintainers and release reviewers | **Authority:** Operational | **Applies to:** Aptus 0.2 | **Owner:** Release engineering | **Last reviewed:** 2026-08-04 | **Review by:** 2026-10-27

Copy this template into a new, dated evidence record for each release candidate.
Do not edit the template into a claim that work passed. Fill every applicable
field with links, digests, commands, and results. Mark a gate `Not run`,
`Blocked`, or `Failed` when evidence is absent.

Passing repository tests is not evidence that a training path works on its
target runtime and hardware.

## Record identity

| Field | Value |
|---|---|
| Release candidate | `[fill: version or tag]` |
| Repository commit | `[fill: full commit hash]` |
| Evidence record date | `[fill: ISO 8601 UTC]` |
| Evidence owner | `[fill: name or role]` |
| Independent reviewer | `[fill: name or role]` |
| Source checkout status | `[fill: clean or explain differences]` |
| CI run | `[fill: immutable URL and result]` |
| Scope | `[fill: methods, placements, platforms, and exclusions]` |
| Overall decision | `[fill: Pass, Fail, Blocked, or Not ready]` |

## Claim boundary

State exactly what this record supports:

`[fill: bounded release claim]`

State what it does not support:

- `[fill: untested method, model family, placement, platform, or artifact]`
- `[fill: absent quality, safety, throughput, cost, or deployment claim]`

## Source and packaging

| Gate | Command or evidence | Result | Evidence location |
|---|---|---|---|
| Clean checkout | `[fill]` | `[fill]` | `[fill]` |
| Python 3.11 test matrix | `[fill]` | `[fill]` | `[fill]` |
| Python 3.12 test matrix | `[fill]` | `[fill]` | `[fill]` |
| Python compile check | `[fill]` | `[fill]` | `[fill]` |
| Ruff lint and format | `[fill]` | `[fill]` | `[fill]` |
| Web tests | `[fill]` | `[fill]` | `[fill]` |
| Generated TypeScript contract | `npm run openapi:check` | `[fill]` | `[fill]` |
| Web type check | `[fill]` | `[fill]` | `[fill]` |
| Web production build | `[fill]` | `[fill]` | `[fill]` |
| Generated OpenAPI contract | `python tools/generate_openapi.py --check` | `[fill]` | `[fill]` |
| Maintained client boundary | `python tools/check_client_contracts.py` | `[fill]` | `[fill]` |
| Version parity | `python tools/verify_versions.py` | `[fill]` | `[fill]` |
| Production dependency audit | `npm audit --omit=dev` | `[fill]` | `[fill]` |
| Full development dependency audit | `npm audit` | `[fill]` | `[fill advisories and disposition]` |
| Wheel build | `[fill]` | `[fill]` | `[fill]` |
| Installed-wheel CLI/API/asset smoke | `[fill]` | `[fill]` | `[fill]` |
| Package-free snapshot/evaluator smoke | `[fill: copied plan_contract.py and policy_snapshot.py with no installed Aptus import]` | `[fill]` | `[fill]` |
| Documentation checks | `[fill]` | `[fill]` | `[fill]` |
| Patch whitespace | `[fill]` | `[fill]` | `[fill]` |

Record built artifacts:

| Artifact | Path or immutable URL | SHA-256 | Size |
|---|---|---|---:|
| Wheel | `[fill]` | `[fill]` | `[fill]` |
| Source distribution, if published | `[fill]` | `[fill]` | `[fill]` |
| Packaged workbench index | `[fill]` | `[fill]` | `[fill]` |
| `Aptus.app` ZIP | `[fill]` | `[fill]` | `[fill]` |
| `Aptus-macOS-arm64.dmg` | `[fill]` | `[fill]` | `[fill]` |
| `COMMIT` source marker | `[fill]` | `[fill]` | `[fill]` |
| `SHA256SUMS` | `[fill]` | `[fill]` | `[fill]` |
| Repeated-gate ledger | `[fill]` | `[fill]` | `[fill]` |
| Repeated-gate log archive | `[fill]` | `[fill]` | `[fill]` |

### Desktop distribution evidence

| Gate | Required evidence | Result |
| --- | --- | --- |
| Source identity | `COMMIT` equals the tested clean checkout | `[fill]` |
| Ten-build stability | Ten consecutive complete builds, durations, artifact hashes, and full logs | `[fill]` |
| Product tests per build | Python, generated contracts, React, typecheck, production web build, and native tests | `[fill]` |
| Packaged launch | Authenticated backend readiness, React readiness, and clean session shutdown | `[fill]` |
| Architecture and identity | arm64 app and backend, bundle ID, version, and build | `[fill]` |
| Application ZIP | Archive integrity, permission preservation, extraction, and strict signature verification | `[fill]` |
| DMG | Creation, `hdiutil verify`, install, and launch result | `[fill]` |
| Developer ID | Identity, Team ID, hardened runtime, timestamp, and nested signatures | `[fill or Not run]` |
| Notarization | App and DMG submissions, request IDs, accepted status, and logs | `[fill or Not run]` |
| Stapling and Gatekeeper | App and DMG staple validation plus `spctl` assessment | `[fill or Not run]` |
| Exact-head CI | Immutable workflow URL, commit marker, artifact name, checksums, and retention | `[fill]` |

Ad-hoc signing can satisfy local engineering review only. A public distribution
row passes only with Developer ID signing, accepted notarization, stapling, and
Gatekeeper assessment for the exact release artifacts.

## Planner and compiler matrix

Record the exact registry and planner facts from the candidate build:

| Item | Expected | Observed | Result |
|---|---|---|---|
| Method descriptor schema | `aptus.method-descriptor.v1` | `[fill]` | `[fill]` |
| Total descriptors | 11 | `[fill]` | `[fill]` |
| Selectable IDs | `full`, `lora`, `int8-lora`, `qlora` | `[fill]` | `[fill]` |
| Experimental IDs | `dora`, `bitfit`, `adalora`, `sharelora` | `[fill]` | `[fill]` |
| Research-only IDs | `loreft`, `aflora`, `bilora` | `[fill]` | `[fill]` |
| Planner rows | 12 | `[fill]` | `[fill]` |
| Plan schema | `aptus.training-plan.v5` | `[fill]` | `[fill]` |
| Policy snapshot schema and digest | `aptus.model-policy-snapshot.v1` | `[fill]` | `[fill]` |
| Model policy decision | `aptus.model-compatibility.v2` | `[fill]` | `[fill]` |
| Inspection receipt | `aptus.model-inspection-receipt.v1` or explicit null | `[fill]` | `[fill]` |
| Candidate policy binding | `aptus.model-policy-binding.v1` on exact path only | `[fill]` | `[fill]` |
| Memory formula | `aptus-memory-v2` | `[fill]` | `[fill]` |
| MLX memory formula | `aptus-memory-mlx-v2` | `[fill]` | `[fill]` |
| Bundle schema | `aptus.bundle.v3` | `[fill]` | `[fill]` |

Attach evidence for:

- [ ] selectable registry IDs exactly equal the executable enum;
- [ ] nonselectable descriptors have no compiler or export contract;
- [ ] all evidence IDs resolve;
- [ ] all aliases and compiler IDs are unique;
- [ ] all 12 placement rows appear with correct status and reasons;
- [ ] every candidate links to the same policy decision, and only an exact
      registered path carries a non-null binding;
- [ ] provider-inspection and user-attested plan sources enforce their receipt
      presence rules;
- [ ] compatibility-subject and observed-planning-facts digests are recomputed
      independently, while parameters and training permission remain outside
      the receipt;
- [ ] malformed, stale, mismatched, and modified receipts fail without
      downgrading to user-attested;
- [ ] the canonical snapshot file, plan `model_policy_snapshot_sha256`, manifest
      `policy_snapshot_sha256`, manifested file digest, and observed current-host
      digest are recorded; every digest has exact lowercase SHA-256 shape;
- [ ] host and portable evaluators return identical complete decisions for the
      exact, near-match, dense, sparse, unknown, and unsorted multi-error subject
      cases;
- [ ] all six `POLICY_SNAPSHOT_*` findings are exercised, including null,
      malformed, noncanonical, wrong-path, invalid-binding, and stale-host cases;
- [ ] package-free validation succeeds against its intact frozen snapshot after
      a simulated host-policy change, while installed-host submission, pilot
      authorization, worker launch, recovery, and the completion verification
      and promotion transaction reject non-current policy;
- [ ] v4, v3, v2, schema-less, and stale-policy or stale-snapshot v5 plans return
      `replan_required` without changing saved bytes;
- [ ] identity mutation tests pass;
- [ ] memory component and upper-bound arithmetic passes;
- [ ] compilation and ZIP output are deterministic and no-clobber;
- [ ] source mutation, path tamper, symlink, and manifest tamper fail;
- [ ] every supported source row is canonicalized, not only the profile sample;
- [ ] Apple Silicon discovery records shared unified memory without inventing
      dedicated VRAM, and live available memory constrains MLX planning.
- [ ] MLX-LM LoRA and QLoRA runtime contracts compile only as conditional,
      single-device paths; PyTorch MPS remains compilerless.
- [ ] The exact Qwen3 MoE row binds `qwen3_moe`,
      `Qwen3MoeForCausalLM`, four-bit checkpoint metadata, a complete
      group-64 layout with exactly one eight-bit group-64 router-gate override
      per layer, a complete no-shared-expert topology, MLX-LM QLoRA, single
      placement, and attention-only targets.

## Target-host inventory

Create one subsection per physical or isolated target-host configuration.

### Host `[fill: host evidence ID]`

| Field | Value |
|---|---|
| Operating system and kernel | `[fill]` |
| Python | `[fill]` |
| Training runtime, compiler, and estimator IDs | `[fill]` |
| CUDA runtime and driver | `[fill]` |
| MLX and MLX-LM versions, when applicable | `[fill]` |
| GPU model, visible index, UUID, and compute capability | `[fill]` |
| Total and pre-run free VRAM by device | `[fill]` |
| Apple chip, unified memory, live available memory, pressure, and swap | `[fill]` |
| Host RAM and pre-run free RAM | `[fill]` |
| Free disk and filesystem | `[fill]` |
| Interconnect | `[fill]` |
| Isolation and unrelated processes | `[fill]` |
| Hardware binding digest | `[fill]` |
| Probe command and raw output location | `[fill]` |

Record whether this host is dedicated during each pilot. Aptus does not reserve
resources against unrelated processes.

## Model and dataset fixtures

Use immutable, reviewable fixtures for every runtime row.

| Fixture ID | Model repository and commit | Family | License and permission evidence | Dataset digest and schema | Sequence and batch | Purpose |
|---|---|---|---|---|---|---|
| `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` |

For every provider-inspected fixture, record:

| Field | Value |
|---|---|
| Receipt ID and schema | `[fill]` |
| Resolved immutable revision | `[fill]` |
| Decision ID and `subject_facts_sha256` | `[fill]` |
| `observed_facts_sha256` and covered field list | `[fill]` |
| `model_policy_snapshot_sha256` | `[fill]` |
| Decision source | `[fill: provider-inspection or user-attested]` |
| Policy ID and version | `[fill or N/A]` |
| Matched path ID | `[fill or N/A]` |
| Receipt exclusion check | `[fill: parameters and training permission absent]` |

Treat these values as tamper-evident content bindings, not authenticated
signatures. Record the trusted local producer and review boundary separately.

For each dataset record:

- source path or protected artifact identity;
- SHA-256 and byte size;
- schema counts and canonical row count;
- split-group policy and declared-group count;
- rights, consent, privacy, and retention review;
- exact test data kept outside training;
- any permitted release-fixture publication limits.

Never place private dataset content in a public evidence record.

## Runtime support matrix

Complete one row for every claimed compiler path and placement. Unsupported rows
need negative-test evidence, not a training pilot.

The first table records `transformers-peft-cuda` paths.

| Method | Placement | Catalog status | Host ID | Plan ID | Candidate ID | Bundle fingerprint | Dependency | Model-data | Preflight | Pilot | Full run | Decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Full | Single | Gated executable with BF16 | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` |
| Full | DDP | Gated executable with BF16 | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` |
| Full | FSDP | Unsupported | N/A | `[fill]` | `[fill]` | N/A | N/A | N/A | N/A | N/A | N/A | `[fill negative evidence]` |
| LoRA | Single | Gated executable | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` |
| LoRA | DDP | Gated executable | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` |
| LoRA | FSDP | Conditional | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` |
| int8-LoRA | Single | Gated executable | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` |
| int8-LoRA | DDP | Gated executable | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` |
| int8-LoRA | FSDP | Unsupported | N/A | `[fill]` | `[fill]` | N/A | N/A | N/A | N/A | N/A | N/A | `[fill negative evidence]` |
| QLoRA | Single | Gated executable | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` |
| QLoRA | DDP | Gated executable | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` |
| QLoRA | FSDP | Unsupported | N/A | `[fill]` | `[fill]` | N/A | N/A | N/A | N/A | N/A | N/A | `[fill negative evidence]` |

Record the Apple paths separately. A passed row requires the runtime-specific
uninterrupted pilot and, when claimed, a parent-verified full-duration run.

| Runtime | Method | Placement | Catalog status | Host ID | Plan ID | Candidate ID | Dependency | Model-data | Preflight | Pilot | Full run | Decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `mlx-lm` | LoRA | Single | Conditional | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` |
| `mlx-lm` | QLoRA | Single | Conditional after pinned four-bit metadata check | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` |
| `mlx-lm` | QLoRA, exact Qwen3 MoE row | Single | Conditional after identity, topology, and four-bit checks | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` |
| `pytorch-mps` | Any | Any | Implementation required | `[fill]` | `[fill]` | `[fill]` | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` | `[fill negative evidence]` |

## Per-path evidence packet

Repeat this section for each runtime row marked passed.

### Path `[fill: method, placement, host, model fixture]`

#### Identity

| Field | Value |
|---|---|
| Plan ID | `[fill]` |
| Model-policy snapshot SHA-256 | `[fill]` |
| Candidate ID | `[fill]` |
| Bundle fingerprint | `[fill]` |
| Dataset SHA-256 | `[fill]` |
| Model and tokenizer commit | `[fill]` |
| Environment binding | `[fill]` |
| Hardware binding | `[fill]` |
| Provider model type and architecture | `[fill]` |
| Checkpoint precision and quantization-layout digest | `[fill]` |
| MoE topology, when applicable | `[fill: experts, selected experts, expert width, sparse cadence, dense-only layers, shared expert]` |
| Total, active, and sparse-layer counts | `[fill]` |
| Adapter target scope | `[fill]` |

#### Ordered actions

| Action | Job ID | Start and finish | Report state | Log SHA-256 | Result |
|---|---|---|---|---|---|
| Dependency | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` |
| Model-data | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` |
| Measured preflight | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` |
| Pilot | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` |
| Full training | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` |

#### Trainable state

| Check | Observed | Result |
|---|---|---|
| Tensor count | `[fill]` | `[fill]` |
| Parameter count | `[fill]` | `[fill]` |
| Name-shape-dtype digest | `[fill]` | `[fill]` |
| Method scope | `[fill]` | `[fill]` |
| Complete target-instance pairs, when applicable | `[fill]` | `[fill]` |
| Exact optimizer membership | `[fill]` | `[fill]` |
| CUDA pilot-phase equality | `[fill]` | `[fill]` |

#### Resource and runtime evidence

For CUDA, record the two checkpoint-continuation phases:

| Metric | Phase 1 | Phase 2 | Full run |
|---|---:|---:|---:|
| Peak runtime memory bytes by rank or MLX process | `[fill]` | `[fill]` | `[fill]` |
| Free VRAM or live unified-memory headroom at admission | N/A | N/A | `[fill]` |
| Free host RAM at admission | N/A | N/A | `[fill]` |
| Free disk at admission | N/A | N/A | `[fill]` |
| Completed steps | `[fill]` | `[fill]` | `[fill]` |
| Finite loss | `[fill]` | `[fill]` | `[fill]` |
| Checkpoint bytes | `[fill]` | `[fill]` | `[fill]` |

- [ ] Phase two started from the expected phase-one step.
- [ ] Checkpoint paths, sizes, hashes, optimizer, scheduler, RNG, scaler where
      applicable, and distributed state passed.
- [ ] `checkpoint_continuation_observed` is true.

For MLX-LM, record the uninterrupted pilot separately:

| Metric | Pilot | Full run |
|---|---:|---:|
| Completed optimizer updates | `[fill, at least 2]` | `[fill, at least 1]` |
| Finite train and validation loss | `[fill]` | `[fill]` |
| Exact target-instance census digest | `[fill]` | `[fill]` |
| Positive adapter delta | `[fill]` | `[fill]` |
| Measured peak MLX bytes | `[fill]` | `[fill]` |
| Available unified memory and required reserve | `[fill]` | `[fill]` |
| Owned artifact-manifest digest | `[fill]` | `[fill]` |
| Fresh-process generated token count, 1 to 4 | `[fill]` | `[fill]` |
| Adapter reload evidence digest | `[fill]` | `[fill]` |

- [ ] MLX pilot and full runs started from the pinned base and completed without
      interruption.
- [ ] MLX output records `resume_supported: false`.
- [ ] All MLX resume argument paths fail closed.
- [ ] Periodic MLX files are described as weight snapshots, not checkpoints.
- [ ] Qwen3 MoE metrics bind the exact provider identity, complete topology,
      canonical quantization layout and digest, logical total and active
      parameter census, sparse-layer count, and attention-only target instances.

#### Full-run split and export

| Field | Value |
|---|---|
| Run ID and output path | `[fill]` |
| Split strategy | `[fill]` |
| Canonical digest | `[fill]` |
| Assignment digest | `[fill]` |
| Total/train/evaluation rows | `[fill]` |
| Declared groups and split units | `[fill]` |
| Target/realized evaluation rows and error | `[fill]` |
| Rank agreement | `[fill]` |
| Final export schema | `[fill]` |
| Final export manifest digest | `[fill]` |
| Completion attestation digest | `[fill]` |
| Final report state | `[fill]` |

- [ ] No declared group crossed the split.
- [ ] Canonical data remained stable through split and lazy consumption.
- [ ] All ranks agreed on split bindings.
- [ ] Safetensors and index checks passed.
- [ ] Recursive file-manifest coverage passed.
- [ ] Parent promotion was idempotent.

## Required negative evidence

Record test names and immutable results for:

- [ ] full FP16 rejection;
- [ ] full FSDP rejection;
- [ ] int8-LoRA and QLoRA FSDP rejection;
- [ ] unsupported backend and non-SFT rejection;
- [ ] packing and enforced wall-time rejection;
- [ ] mutable or unpinned model revision rejection;
- [ ] missing training permission rejection;
- [ ] source and canonical-data mutation rejection;
- [ ] missing, invalid-JSON, null, malformed-contract, noncanonical, wrong-path,
      invalid-digest-binding, and differing-digest policy-snapshot rejection;
- [ ] package-free frozen-snapshot integrity versus installed-host currency
      behavior, including coherent stale-v5 `replan_required`;
- [ ] installed-host non-object plan, manifest, trainer, and snapshot rejection,
      plus package-free non-object plan, manifest, and snapshot rejection,
      without escaped parser or primitive-shape exceptions on covered readers;
- [ ] empty, extra, non-finite, boolean-typed, malformed, and mismatched census
      rejection;
- [ ] missing LoRA pair and optimizer-set mismatch rejection;
- [ ] checkpoint corruption and continuation mismatch rejection;
- [ ] MLX resume-argument rejection and weight-snapshot non-resume boundary;
- [ ] current capacity regression after pilot rejection;
- [ ] MoE shared-expert, wrong-precision, malformed-topology, wrong-runtime,
      wrong-method, distributed-placement, and expanded-adapter-scope rejection;
- [ ] wrong run, rank, split, metric, and export binding rejection;
- [ ] child exit zero with invalid evidence rejection;
- [ ] full-run resume rejection;
- [ ] concurrent state-root submission rejection;
- [ ] cancellation and stale-owner reconciliation failure paths.

## Job control and recovery

| Scenario | Host ID | Job IDs | Evidence | Result |
|---|---|---|---|---|
| Same-state concurrent submission | `[fill]` | `[fill]` | `[fill]` | `[fill]` |
| Cross-state host-global lease | `[fill]` | `[fill]` | `[fill]` | `[fill]` |
| POSIX process-group cancellation | `[fill]` | `[fill]` | `[fill]` | `[fill]` |
| Cancelling reconciliation | `[fill]` | `[fill]` | `[fill]` | `[fill]` |
| Parent crash before promotion | `[fill]` | `[fill]` | `[fill]` | `[fill]` |
| Verified pending promotion recovery | `[fill]` | `[fill]` | `[fill]` | `[fill]` |
| Host policy changes before worker launch | `[fill]` | `[fill]` | `[fill]` | `[fill]` |
| Host policy changes before pending promotion | `[fill]` | `[fill]` | `[fill]` | `[fill]` |
| Windows managed-path behavior, if claimed | `[fill]` | `[fill]` | `[fill]` | `[fill]` |

## API and workbench

- [ ] Strict schemas reject unknown and resume fields.
- [ ] V5 response normalization rejects a missing, non-string, uppercase, short,
      or non-hexadecimal `model_policy_snapshot_sha256`.
- [ ] Saved-plan load, compile, project recovery, and managed job submission map
      legacy and coherent stale-policy v5 state to structured HTTP
      `409 replan_required`, not generic `400 invalid_request`; host static
      validation records the typed digest finding instead.
- [ ] Bootstrap exposes all 11 descriptors and only four selectable IDs.
- [ ] Runtime validation routes through cancellable jobs.
- [ ] Active jobs guard hardware scan and conflicting actions.
- [ ] Apple memory is labeled shared, live headroom is separate, and eligible
      MLX actions proceed only through their ordered prerequisites.
- [ ] Method preference cannot select experimental or research-only methods.
- [ ] All five runtime actions, phase, output, attestation, and integrity are
      visible.
- [ ] Cached authorization is not presented as current admission.
- [ ] Example mode remains visibly non-executed.
- [ ] Keyboard, focus, live-region, contrast, reduced-motion, and narrow-layout
      checks passed against the packaged build.

Attach browser, automated, and installed-wheel evidence:

`[fill: locations and digests]`

## Security and data handling

- [ ] Secret scan scope and result are recorded.
- [ ] Loopback is the default.
- [ ] Non-loopback behavior requires explicit acknowledgment and remains outside
      the built-in security boundary.
- [ ] Provider metadata cannot set training permission.
- [ ] Cleartext source, canonical, pilot, ZIP, cache, CUDA checkpoint, MLX
      weight-snapshot, log, and export locations were reviewed.
- [ ] Generated source and direct dependency changes received manual review.
- [ ] No private data, token, model weight, checkpoint, or machine artifact is
      included in the public release evidence.

## Documentation consistency

- [ ] README and current-capability claims match the executable matrix.
- [ ] CLI and API references match help and request models.
- [ ] Plan, bundle, validation, run-state, security, and recovery references are
      current.
- [ ] Current pages document all six snapshot findings, every plan-identity
      snapshot binding, and the frozen-integrity versus host-currency boundary.
- [ ] No current page calls direct pins a transitive lock.
- [ ] No current page offers full-training resume or full FSDP.
- [ ] No current page claims guaranteed fit, quality, or universal optimality.
- [ ] Method lifecycle wording matches the runtime registry.
- [ ] Dataset split strategy identifiers and grouped error behavior are current.
- [ ] Apple discovery, uninterrupted MLX adapter execution, MLX no-resume
      semantics, and PyTorch MPS compiler absence are described as distinct
      facts.
- [ ] LM Studio and oMLX remain inference-only.
- [ ] Future evaluation, export, provider, cloud, and MCP work is labeled future.

## Failures, deviations, and open gates

| ID | Severity | Gate | Finding | Evidence | Owner | Required resolution | Status |
|---|---|---|---|---|---|---|---|
| `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` |

Do not omit failed experiments. State whether each failure blocks the release or
narrows its support claim.

## Final review

| Reviewer | Role | Scope reviewed | Decision | Date | Evidence signature or digest |
|---|---|---|---|---|---|
| `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` |

Final decision rationale:

`[fill]`

Remaining unsupported or unproven paths:

- `[fill]`

## Related documentation

- [Release gates](release-gates.md)
- [Operator checklist](operator-checklist.md)
- [Current capabilities](../product/current-capabilities.md)
- [Validation states](../reference/validation-states.md)
- [Security policy](../../SECURITY.md)
