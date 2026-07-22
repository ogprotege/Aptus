# Release Evidence Template

> **Status:** Active template | **Audience:** Maintainers and release reviewers | **Authority:** Operational | **Applies to:** Aptus 0.2 | **Owner:** Release engineering | **Last reviewed:** 2026-07-22 | **Review by:** 2026-10-22

Copy this template into a new, dated evidence record for each release candidate.
Do not edit the template into a claim that work passed. Fill every applicable
field with links, digests, commands, and results. Mark a gate `Not run`,
`Blocked`, or `Failed` when evidence is absent.

Passing repository tests on a non-CUDA development host is not evidence that a
training path works on its target hardware.

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
| Web type check | `[fill]` | `[fill]` | `[fill]` |
| Web production build | `[fill]` | `[fill]` | `[fill]` |
| Wheel build | `[fill]` | `[fill]` | `[fill]` |
| Installed-wheel CLI/API/asset smoke | `[fill]` | `[fill]` | `[fill]` |
| Documentation checks | `[fill]` | `[fill]` | `[fill]` |
| Patch whitespace | `[fill]` | `[fill]` | `[fill]` |

Record built artifacts:

| Artifact | Path or immutable URL | SHA-256 | Size |
|---|---|---|---:|
| Wheel | `[fill]` | `[fill]` | `[fill]` |
| Source distribution, if published | `[fill]` | `[fill]` | `[fill]` |
| Packaged workbench index | `[fill]` | `[fill]` | `[fill]` |

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
| Plan schema | `aptus.training-plan.v2` | `[fill]` | `[fill]` |
| Memory formula | `aptus-memory-v2` | `[fill]` | `[fill]` |
| Bundle schema | `aptus.bundle.v2` | `[fill]` | `[fill]` |

Attach evidence for:

- [ ] selectable registry IDs exactly equal the executable enum;
- [ ] nonselectable descriptors have no compiler or export contract;
- [ ] all evidence IDs resolve;
- [ ] all aliases and compiler IDs are unique;
- [ ] all 12 placement rows appear with correct status and reasons;
- [ ] identity mutation tests pass;
- [ ] memory component and upper-bound arithmetic passes;
- [ ] compilation and ZIP output are deterministic and no-clobber;
- [ ] source mutation, path tamper, symlink, and manifest tamper fail;
- [ ] every supported source row is canonicalized, not only the profile sample;
- [ ] Apple Silicon discovery remains inventory-only and nonexecutable.

## Target-host inventory

Create one subsection per physical or isolated target-host configuration.

### Host `[fill: host evidence ID]`

| Field | Value |
|---|---|
| Operating system and kernel | `[fill]` |
| Python | `[fill]` |
| CUDA runtime and driver | `[fill]` |
| GPU model, visible index, UUID, and compute capability | `[fill]` |
| Total and pre-run free VRAM by device | `[fill]` |
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

## Per-path evidence packet

Repeat this section for each runtime row marked passed.

### Path `[fill: method, placement, host, model fixture]`

#### Identity

| Field | Value |
|---|---|
| Plan ID | `[fill]` |
| Candidate ID | `[fill]` |
| Bundle fingerprint | `[fill]` |
| Dataset SHA-256 | `[fill]` |
| Model and tokenizer commit | `[fill]` |
| Environment binding | `[fill]` |
| Hardware binding | `[fill]` |

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
| Pilot-phase equality | `[fill]` | `[fill]` |

#### Resource and continuation evidence

| Metric | Phase 1 | Phase 2 | Full run |
|---|---:|---:|---:|
| Peak CUDA bytes by rank | `[fill]` | `[fill]` | `[fill]` |
| Free VRAM at admission | N/A | N/A | `[fill]` |
| Free host RAM at admission | N/A | N/A | `[fill]` |
| Free disk at admission | N/A | N/A | `[fill]` |
| Completed steps | `[fill]` | `[fill]` | `[fill]` |
| Finite loss | `[fill]` | `[fill]` | `[fill]` |
| Checkpoint bytes | `[fill]` | `[fill]` | `[fill]` |

- [ ] Phase two started from the expected phase-one step.
- [ ] Checkpoint paths, sizes, hashes, optimizer, scheduler, RNG, scaler where
      applicable, and distributed state passed.
- [ ] `checkpoint_continuation_observed` is true.

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
- [ ] empty, extra, non-finite, boolean-typed, malformed, and mismatched census
      rejection;
- [ ] missing LoRA pair and optimizer-set mismatch rejection;
- [ ] checkpoint corruption and continuation mismatch rejection;
- [ ] current capacity regression after pilot rejection;
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
| Windows managed-path behavior, if claimed | `[fill]` | `[fill]` | `[fill]` | `[fill]` |

## API and workbench

- [ ] Strict schemas reject unknown and resume fields.
- [ ] Bootstrap exposes all 11 descriptors and only four selectable IDs.
- [ ] Runtime validation routes through cancellable jobs.
- [ ] Active jobs guard hardware scan and conflicting actions.
- [ ] Apple inventory is labeled shared memory and nonexecutable.
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
- [ ] Cleartext source, canonical, pilot, ZIP, cache, checkpoint, log, and export
      locations were reviewed.
- [ ] Generated source and direct dependency changes received manual review.
- [ ] No private data, token, model weight, checkpoint, or machine artifact is
      included in the public release evidence.

## Documentation consistency

- [ ] README and current-capability claims match the executable matrix.
- [ ] CLI and API references match help and request models.
- [ ] Plan, bundle, validation, run-state, security, and recovery references are
      current.
- [ ] No current page calls direct pins a transitive lock.
- [ ] No current page offers full-training resume or full FSDP.
- [ ] No current page claims guaranteed fit, quality, or universal optimality.
- [ ] Method lifecycle wording matches the runtime registry.
- [ ] Dataset split strategy identifiers and grouped error behavior are current.
- [ ] Apple discovery remains distinct from MPS or MLX execution.
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
