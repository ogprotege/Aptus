# M0 Path Beta identity freeze

> **Status:** Frozen for program duration (M0.2)  
> **Path ID:** `path-beta-cuda-lora-single-v1`  
> **Frozen:** 2026-08-11  
> **Authority:** Mission integrity plan Phase M0; historical measured evidence only — not a current-HEAD re-proof

## Recommendation

**Prefer SmolLM2-135M-Instruct LoRA single-device** already measured on the RTX 3050 campaign/acceptance host class.

**Primary freeze = August 6 acceptance identity** — the only packet that records exact-scope `measured-run-pass` with the full five-job ladder (dependency → model-data → preflight → pilot with checkpoint continuation → full train), parent-verified structural PEFT export, and bound plan/candidate/bundle/policy digests.

**Phase 5 (2026-08-10)** is **supporting repeatability evidence** for the same model class (SmolLM2-135M-Instruct, immutable revision `12fd25f7…`, LoRA, BF16, single placement, world size 1, Ubuntu RTX 3050 host class). It is **not** merged into the primary identity: Phase 5 binds a **different source commit/tree**, a **different dataset fixture/digest**, and a campaign protocol cell (128 optimizer steps, five slots) rather than the acceptance five-job managed workflow. Treat them as related evidence under separate labels.

Phase 10 certification aggregates campaign cells and does **not** replace or redefine the primary acceptance identity.

---

## Primary identity table

One primary freeze only. Values are copied from the August 6 acceptance packet; no hash was invented.

| Field | Value |
| --- | --- |
| Path ID | `path-beta-cuda-lora-single-v1` |
| Training runtime | `transformers-peft-cuda` |
| Method | `lora` |
| Placement | `single` (world size 1, CUDA device 0) |
| Model repo | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| Immutable revision | `12fd25f77366fa6b3b4b768ec3050bf629380bac` |
| Architecture / params if recorded | `LlamaForCausalLM`, 134,515,008 parameters |
| Dataset path | `examples/support-sft.jsonl` (four synthetic contract rows) |
| Dataset SHA-256 | `bf2dca3d6398d639f47a883203920e1f52b0981becac96734147054e53f8aa44` |
| Host class | Ubuntu 24.04.4 LTS + NVIDIA GeForce RTX 3050 (~8 GB class; Torch-visible VRAM 8,220,573,696 bytes in acceptance) |
| Precision notes | BF16 LoRA; compiler `transformers.peft-lora.v2`; export `peft-adapter-safetensors` |
| Historical acceptance source commit | `c12c4d8db0037a2c278a2ad95a0a2cbda4387eed` (tree `ad482883cfb6ad2b8ac72f7b7d1009c918e5c345`) |
| Bundle fingerprint | `296fb7b710f60345a590748f053eb15f9b5b4f4b3fec539ae3a705e31d6a640b` |
| Policy snapshot SHA-256 if recorded | `c2ae989c8b68df6e984dc7c8670397e791ff30e1f5ce82129e25c1c2b93268d8` |
| Success state | `measured-run-pass` |
| Evidence packet path(s) | `docs/operations/evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/` |

### Acceptance bindings retained with the primary identity (not a second freeze)

| Binding | Value |
| --- | --- |
| Plan ID | `plan_ed1d3fc5ecef48acf6af` |
| Candidate ID | `cand_e0f49f5d708a27ff96ca` |
| Bundle ZIP SHA-256 | `6e79b01f5b723ef9b30f7bc5f886fa5e93e2884ba504733b9d8194ef5c5a04c1` |
| Export verification | Structural PEFT adapter tree rehash; `adapter_model.safetensors` SHA-256 `fd3eb151acf70ab072eb8a60186df782370fa182b74dd92f8630591ba7a9dba5` (19,593,064 bytes) |
| Qualifying jobs | Exactly five managed jobs, all return code 0: dependency, model/data, preflight, pilot, full train |
| Runtime closure (historical) | Python 3.12.3, Torch 2.13.0+cu130, Transformers 5.14.1, Accelerate 1.14.0, Safetensors 0.8.0, PEFT 0.19.1; bitsandbytes not installed |

---

## Related evidence

These packets reinforce LoRA single-device stability on the same host class / model class. They **do not** replace the primary identity and **must not** be merged with the acceptance source commit into one unlabeled identity.

| Packet | Role | One-line claim boundary |
| --- | --- | --- |
| `docs/operations/evidence/2026-08-10-cuda-phase5-repeatability-anchor/` | Supporting repeatability for SmolLM2-135M LoRA single BF16 on RTX 3050 | Exact-host five-slot repeatability + duration/peak-memory stability for frozen Phase 5 cohort (source `3bfec547d4cffedbaf049426d9713f1ccc25b5a2`); not model quality, not production safety, not release readiness, not another method/placement/host |
| `docs/operations/evidence/2026-08-11-cuda-phase10-certification/` | Campaign aggregation / certification of prior phases | Aggregates Phase 5–9 bounded campaign cells with independent review; no new training; not Aptus 0.2 release readiness, not quality, not semantic CUDA adapter reload, not DDP/LoRA FSDP, not multi-GPU |

### Phase 5 vs primary — labeled differences (do not merge)

| Aspect | Primary (acceptance 2026-08-06) | Phase 5 repeatability (2026-08-10) |
| --- | --- | --- |
| Source commit | `c12c4d8db0037a2c278a2ad95a0a2cbda4387eed` | `3bfec547d4cffedbaf049426d9713f1ccc25b5a2` |
| Source tree | `ad482883cfb6ad2b8ac72f7b7d1009c918e5c345` | `6acaa096ad50b0e814e84e706d3dd12a3cc8cc33` |
| Model revision | `12fd25f77366fa6b3b4b768ec3050bf629380bac` (same) | same |
| Method / placement / precision | LoRA, single, BF16 | LoRA, single, BF16 |
| Dataset | `examples/support-sft.jsonl` SHA-256 `bf2dca3d…` | Campaign fixture SHA-256 `6d90599e949bf2698b940e0c159e1fa24f3dc0c162005546bd270fc761aac7f2` |
| Success framing | Managed five-job `measured-run-pass` + parent promotion | Five protocol-valid native passes, 128 optimizer steps each |
| Bundle fingerprint | Bound (`296fb7b7…`) | Not used as primary Path Beta bundle identity |

---

## Explicit non-claims

This freeze does **not** establish or authorize:

- DDP, FSDP, LoRA FSDP, or any multi-GPU placement
- Compatibility with all CUDA cards, drivers, or VRAM classes beyond the bound host class
- Full, int8-LoRA, QLoRA, or other method matrix cells as Path Beta
- Model quality, safety, convergence, training rights, or benchmark/throughput guarantees
- Semantic adapter reload / fresh-process generation check for CUDA PEFT adapters (acceptance verification was structural only; Phase 10 explicitly leaves semantic CUDA adapter reload open)
- Current-HEAD re-proof — historical acceptance and Phase 5 commits are **not** HEAD; M4 must re-prove
- Production readiness or Aptus release readiness from this freeze alone
- Full-run resume beyond the acceptance pilot’s two-phase checkpoint continuation observation
- Merging acceptance and Phase 5 into a single unlabeled “proven” identity

---

## Program implication

- **M4 (Path Beta release-honest)** must re-prove this identity (or a deliberately re-bound successor identity documented under Section 12) at **then-current HEAD** on an **authorized** CUDA host of the frozen host class (or a recorded host-class change decision).
- Clean-env dependency install is required by the release gates / ordered job ladder; do not inherit an impure training interpreter as if it were the historical acceptance closure.
- Historical digests (bundle fingerprint, policy snapshot, acceptance commit) are **anchors for comparison and claim language**, not a license to skip re-measurement when source, deps, model, dataset, or host binding change.
- Prefer reuse of the SmolLM2-135M-Instruct revision and LoRA single BF16 shape unless rights or hardware block it; document any identity change before measured work.

---

## Sources

1. `docs/operations/evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/README.md` — primary measured-run-pass acceptance  
2. `docs/operations/evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/acceptance-summary.json` — machine-readable bindings (runtime, plan, bundle, policy, dataset, host)  
3. `docs/operations/evidence/2026-08-10-cuda-phase5-repeatability-anchor/README.md` — supporting five-slot repeatability  
4. `docs/operations/evidence/2026-08-10-cuda-phase5-repeatability-anchor/phase5-outcome.json` — Phase 5 bindings (source, model revision, dataset fixture digest, configuration)  
5. `docs/operations/evidence/2026-08-11-cuda-phase10-certification/README.md` — campaign certification claim boundaries only  
6. `docs/product/mission-integrity-plan.md` §7.5 M0.3 / §11 M4 — Path Beta path ID and program requirements  

---

## Self-check

| Check | Result |
| --- | --- |
| Primary is LoRA single, not Full | Yes |
| Primary is single-device, not multi-GPU | Yes |
| Prefer acceptance for first Beta freeze | Yes — August 6 `measured-run-pass` |
| Phase 5 labeled separately (different source commit) | Yes |
| No invented hashes | Yes — all digests copied from listed evidence packets |
| No measured runs / no `src/` edits for this task | Yes (documentation freeze only) |
| Explicit non-claims include quality, DDP/FSDP, current-HEAD | Yes |
| M4 re-proof + clean-env requirement stated | Yes |
