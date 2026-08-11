# M1 — Journey paper walks and evidence packet index

> **Phase:** M1 Promise audit (M1.2 + M1.3)  
> **Mode:** Read-only docs audit (no code, no measured runs)  
> **Authored:** 2026-08-11  
> **Bases:** `M0-PATH-ALPHA-FREEZE.md`, `M0-PATH-BETA-FREEZE.md`  
> **Doc sources:** `docs/product/user-workflows.md`, `docs/product/current-capabilities.md`, `docs/product/mission-integrity-plan.md` §4.2, `docs/getting-started/*`, `docs/guides/compile-validate-run.md`, `docs/operations/operator-checklist.md`, `docs/operations/apple-silicon-pilot.md`, `README.md`, path freezes under this directory

## Status legend

| Tag | Meaning for this paper walk |
| --- | --- |
| `works` | Current product docs describe the step, and capability/operator docs present it as an available surface; an operator can execute it by following published guidance |
| `partial` | Step is real but incomplete for the frozen path identity (scattered docs, tutorial uses a different model/host, historical HEAD only, claim narrower than full mission ideal) |
| `missing` | Required for a release-honest solo journey; dedicated artifact or guidance does not exist yet |
| `unknown` | Docs alone do not establish whether the step succeeds for the frozen identity |

---

# Part A — Journey paper walks

## Journey A — Path Alpha (`path-alpha-mlx-qlora-v1`)

**Target identity (frozen):** MLX-LM · QLoRA · single · `mlx-community/Qwen2.5-0.5B-Instruct-4bit` @ `53a32aee5e9447773fd2b85988395066aef3700a` · dataset `examples/support-sft.jsonl` SHA-256 `bf2dca3d…` · host class Apple M5 Pro / 64 GiB unified · historical success `measured-run-pass` at source `719255153e3fc7e38e83b5ff826d587e5e58bf80` · primary evidence `docs/operations/evidence/2026-08-05-qwen2-mlx-lm-exact-source-refresh/`

**Operator surfaces used in walk:** Mac app (preferred) or `aptus serve` + CLI; external MLX Python selected via Models / doctor.

| # | Step | How docs say to do it | Status | Notes |
| ---: | --- | --- | --- | --- |
| A1 | Install Aptus control plane | Dev: `pip install -e '.[server,test]'` ([install](../../../docs/getting-started/install.md), [README](../../../README.md)). Or build Mac app: `desktop/macos/build.sh` then open `Aptus.app`. | **works** | Ad-hoc signed local app only; public notarization open (not required for personal Alpha). |
| A2 | Create external MLX training env | Separate venv; pin `mlx==0.31.2` and `mlx-lm==0.31.3` ([install §Configure MLX-LM](../../../docs/getting-started/install.md)). | **works** | Sidecar does not absorb training stack; pins match freeze. |
| A3 | Select exact MLX Python (no silent install) | Mac **Models → Choose MLX Python** / environment doctor; CLI `aptus doctor` ([user-workflows](../../../docs/product/user-workflows.md), [operator-checklist](../../../docs/operations/operator-checklist.md)). | **works** | Doctor installs nothing; probe must pass pinned versions. |
| A4 | Profile Path Alpha dataset | `aptus profile --dataset ./examples/support-sft.jsonl …` ([first-plan](../../../docs/getting-started/first-plan.md), [quickstart](../../../docs/getting-started/quickstart.md)). | **works** | Four synthetic rows; digest must remain `bf2dca3d6398d639f47a883203920e1f52b0981becac96734147054e53f8aa44`. |
| A5 | Capture local Apple hardware facts | Workbench **scan this Mac** or `aptus inspect hardware`; `mps` = shared unified memory, not VRAM ([choose-your-path Path A/B](../../../docs/getting-started/choose-your-path.md)). | **works** | MLX estimator uses live headroom when present. |
| A6 | Pin model revision + attest training permission | Model ID + immutable 40–64 hex revision; license label; explicit training-allowed confirmation ([model-dataset-hardware](../../../docs/guides/model-dataset-hardware.md), freeze table). | **works** | Exact freeze artifact: `mlx-community/Qwen2.5-0.5B-Instruct-4bit` @ `53a32aee5e9447773fd2b85988395066aef3700a`. |
| A7 | Optional provider inspection | `aptus inspect model --model-id … --revision …`; review before copy; permission remains user-attested ([user-workflows](../../../docs/product/user-workflows.md)). | **works** | Receipt path preferred for Path Alpha policy binding. |
| A8 | Plan MLX-LM QLoRA single for frozen identity | Facts → plan via workbench or `aptus spec-plan` / API; select MLX-LM, QLoRA, single ([user-workflows Plan](../../../docs/product/user-workflows.md), [choose-your-path Path B](../../../docs/getting-started/choose-your-path.md)). | **partial** | Planning surface **works** in product. **Gap:** no identity-bound Path Alpha plan recipe. `first-plan.md` uses a fake CUDA tutorial model. Dedicated runbook `docs/guides/path-alpha-mlx-operator.md` is **M3**, not present. Operator must assemble freeze + general guides. |
| A9 | Compare candidates + policy records | Review feasible/conditional/rejected; three model-policy records (match / path / evidence readiness); expect policy `model.qwen2-24l.mlx-qlora` v1.0.0 path `mlx-lm.qlora.single.dense-causal-lm.v1` ([current-capabilities](../../../docs/product/current-capabilities.md), freeze). | **works** | Configuration footprint ≠ every Qwen2 artifact accepted at runtime. |
| A10 | Compile no-clobber bundle | `aptus compile --plan … --output …` → `aptus.bundle.v3` + ZIP ([compile-validate-run](../../../docs/guides/compile-validate-run.md)). | **works** | Destination must be empty; never overwrite. |
| A11 | Static validation | `aptus validate BUNDLE --level static` or package-free `validate.py --level static` → `static-pass` ([first-plan](../../../docs/getting-started/first-plan.md), [operator-checklist](../../../docs/operations/operator-checklist.md)). | **works** | Package-free path checks frozen snapshot parity, not host-registry currency. |
| A12 | Dependency action | `aptus run BUNDLE --action dependency` → `dependency-pass` ([choose-your-path Path B](../../../docs/getting-started/choose-your-path.md) step 7, [operator-checklist](../../../docs/operations/operator-checklist.md)). | **works** | Verifies pinned MLX / MLX-LM on selected interpreter. |
| A13 | Model-data action | `aptus run BUNDLE --action model-data` → `model-data-pass` (load pinned 4-bit revision; tokenize all bound rows; four-bit metadata gate). | **works** | Documented MLX QLoRA requirement; historical Alpha ladder passed. |
| A14 | Measured preflight | `aptus run BUNDLE --action preflight` → `measured-preflight-pass` (bounded real MLX adapter smoke + runtime-neutral memory). | **works** | Ordered; skip rejected by managed service. |
| A15 | Pilot (MLX semantics) | `aptus run BUNDLE --action pilot` → `pilot-pass`: uninterrupted ≥2 optimizer updates from pinned base; finite losses; exact targets; fresh-process adapter reload 1–4 tokens ([user-workflows](../../../docs/product/user-workflows.md), [compile-validate-run](../../../docs/guides/compile-validate-run.md)). | **works** | Not CUDA two-phase continuation. No MLX resume. |
| A16 | Confirmed full train → `measured-run-pass` | `aptus run BUNDLE --action train --confirm-full-train`; parent verifies metrics, adapter tree, fresh reload, export; promotes only on success ([operator-checklist](../../../docs/operations/operator-checklist.md)). | **partial** | **Product path works** and is **historically proven** twice at freeze source (`71925515…`, fingerprint `ca2548cf…`) in exact-source refresh packet. **Gaps:** (1) freeze non-claim — not current-HEAD re-proof (M3); (2) no solo Path Alpha runbook; (3) evidence is path-scoped, not product release readiness. |
| A17 | Inspect artifacts; stay inside claim boundary | Review `run_output_dir`, hashes, validation report; `measured-run-pass` = structural/runtime only, not quality ([user-workflows Interpret completion](../../../docs/product/user-workflows.md), freeze non-claims). | **works** | Claim language is explicit in capabilities + freeze. |
| A18 | Complete journey from one Path Alpha operator runbook | Planned `docs/guides/path-alpha-mlx-operator.md` (mission M3.1). | **missing** | End-to-end solo runbook does not exist; journey is stitchable from general docs only. |

### Journey A status counts

| Status | Count |
| --- | ---: |
| works | 15 |
| partial | 2 |
| missing | 1 |
| unknown | 0 |
| **Total steps** | **18** |

**Paper-walk summary:** The product ladder for local MLX-LM QLoRA is documented and historically closed for the exact Path Alpha identity. The release-honest mission gap is **identity-bound operator packaging** (A8/A18) and **current-HEAD re-proof** (A16), not absence of the five managed actions.

---

## Journey B — Path Beta (`path-beta-cuda-lora-single-v1`)

**Target identity (frozen):** `transformers-peft-cuda` · LoRA · single · BF16 · `HuggingFaceTB/SmolLM2-135M-Instruct` @ `12fd25f77366fa6b3b4b768ec3050bf629380bac` · same synthetic dataset digest · host class Ubuntu 24.04 + RTX 3050 (~8 GiB) · historical `measured-run-pass` at source `c12c4d8db0037a2c278a2ad95a0a2cbda4387eed` · primary evidence `docs/operations/evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/` · Phase 5 (2026-08-10) related only (different source commit/dataset fixture)

**Shape:** Mac (or any control plane) plans/compiles → transfer → CUDA host runs ordered gates. Mac never runs CUDA train.

| # | Step | How docs say to do it | Status | Notes |
| ---: | --- | --- | --- | --- |
| B1 | Install Aptus on control plane (Mac) | Same as A1 ([install](../../../docs/getting-started/install.md), [README](../../../README.md)). | **works** | Control-plane only for CUDA. |
| B2 | Profile Path Beta dataset | `aptus profile` on `examples/support-sft.jsonl` ([first-plan](../../../docs/getting-started/first-plan.md)). | **works** | Same digest as Alpha freeze. |
| B3 | Declare CUDA host facts without implying Mac trains CUDA | Manual / user-attested hardware facts for Ubuntu RTX 3050 class, or inspect on the host later ([user-workflows Plan for a different host](../../../docs/product/user-workflows.md), [choose-your-path Path A](../../../docs/getting-started/choose-your-path.md), [current-capabilities](../../../docs/product/current-capabilities.md)). | **works** | Docs explicitly: Mac profile ≠ local CUDA measurement; provenance stays `user-attested` when manual. |
| B4 | Pin SmolLM2 revision + training permission | Exact freeze revision `12fd25f7…`; license + confirm training allowed ([model-dataset-hardware](../../../docs/guides/model-dataset-hardware.md)). | **works** | Prefer inspect on network-capable host/control plane. |
| B5 | Plan CUDA LoRA single BF16 for frozen identity | Workbench or `aptus spec-plan` with `--backend cuda`, LoRA, single placement ([quickstart](../../../docs/getting-started/quickstart.md), [user-workflows](../../../docs/product/user-workflows.md)). | **partial** | CUDA planning **works**. **Gap:** no identity-bound Path Beta plan/compile recipe; `docs/guides/path-beta-cuda-lora-operator.md` is **M4**, not present. README “See it work” uses a 7B Llama tutorial, not SmolLM2 freeze identity. |
| B6 | Compare and select LoRA single | Review all candidates; keep rejects visible; select LoRA single BF16 for Beta ([current-capabilities](../../../docs/product/current-capabilities.md)). | **works** | Full / multi-GPU not Path Beta. |
| B7 | Compile CUDA bundle | `aptus compile` → portable CUDA programs + `requirements.txt` + policy snapshot ([compile-validate-run](../../../docs/guides/compile-validate-run.md)). | **works** | Bundle is the handoff artifact. |
| B8 | Static validation on control plane | `aptus validate … --level static` / package-free static ([operator-checklist Before installing](../../../docs/operations/operator-checklist.md)). | **works** | Does not authorize host train. |
| B9 | Transfer bundle to CUDA host; verify integrity | Copy bundle/ZIP; treat dataset copies as sensitive; verify manifest/fingerprint/snapshot digests on host ([user-workflows](../../../docs/product/user-workflows.md), [choose-your-path Path C](../../../docs/getting-started/choose-your-path.md), freeze bundle fingerprint `296fb7b7…`). | **partial** | Handoff is **described** across docs. **Gap:** no single Path Beta handoff checklist with freeze digests, host prerequisites, and clean-env proof template (M4.1–M4.2). |
| B10 | Clean env outside bundle; install pins | New venv beside (not inside) bundle; install Aptus + `pip install -r requirements.txt` ([compile-validate-run Install the bundle stack](../../../docs/guides/compile-validate-run.md), [operator-checklist](../../../docs/operations/operator-checklist.md)). | **works** | In-bundle venv invalidates manifest. Freeze requires clean-env for M4 re-proof. |
| B11 | Dependency action on host | `aptus run BUNDLE --action dependency` or portable levels ([choose-your-path Path C](../../../docs/getting-started/choose-your-path.md)). | **works** | Historical Beta ladder step 1. |
| B12 | Model-data + trainable census | `… --action model-data` (LoRA A/B pairs only; positive finite census) ([compile-validate-run](../../../docs/guides/compile-validate-run.md)). | **works** | Acceptance recorded 134,515,008 params / 210 adapter targets / 4 rows. |
| B13 | Measured preflight | `… --action preflight` (CUDA synthetic method path + census). | **works** | Ordered gate. |
| B14 | Two-phase real-model pilot | `… --action pilot` (two phases; checkpoint continuation observation for CUDA) ([user-workflows Prove a bundle](../../../docs/product/user-workflows.md)). | **works** | Distinct from MLX pilot. Historical Beta pilot continued steps 1→2. |
| B15 | Review deep admission evidence | Before train: pilot metrics, free VRAM/RAM/disk, policy currency ([compile-validate-run Authorize full training](../../../docs/guides/compile-validate-run.md)). | **works** | Managed admission rechecks registry. |
| B16 | Confirm full train → parent structural export → `measured-run-pass` | `… --action train --confirm-full-train`; parent verifies PEFT tree/hashes; promotes ([operator-checklist](../../../docs/operations/operator-checklist.md), freeze). | **partial** | **Historically works** for exact Aug 6 acceptance (five jobs RC 0, structural adapter SHA bound). **Gaps:** (1) not current-HEAD (M4); (2) **semantic CUDA adapter reload not claimed** (freeze + Phase 10); (3) structural export ≠ quality; (4) Phase 5 repeatability is related, not merged into primary identity. |
| B17 | Return evidence; refuse broader claims | Keep host/cell scope; do not claim all CUDA cards, DDP/FSDP, quality, or release readiness (freeze non-claims, [current-capabilities](../../../docs/product/current-capabilities.md)). | **works** | Claim boundaries published; operator still must obey them. |
| B18 | Complete journey from one Path Beta operator runbook | Planned `docs/guides/path-beta-cuda-lora-operator.md` (mission M4.1). | **missing** | No end-to-end Mac→host Beta runbook. |

### Journey B status counts

| Status | Count |
| --- | ---: |
| works | 14 |
| partial | 3 |
| missing | 1 |
| unknown | 0 |
| **Total steps** | **18** |

**Paper-walk summary:** CUDA single-device LoRA handoff is a real, documented product path and was measured once for the freeze identity. Mission integrity gaps are the **missing Beta runbook**, **handoff packaging**, **clean-env re-proof at HEAD**, and the intentional **structural-only** export claim (no semantic reload).

---

# Part B — Evidence packet index

Top-level entries under `docs/operations/evidence/`. One-line claim boundary paraphrases each packet’s published scope (not an expansion of claims). Tags:

| Tag | Meaning |
| --- | --- |
| `supports_path_alpha` | Directly anchors or baselines Path Alpha MLX Qwen2.5 QLoRA identity |
| `supports_path_beta` | Directly anchors or supports Path Beta SmolLM2 LoRA single acceptance / freeze-listed related Beta anchors |
| `supporting_related` | Same RTX 3050 campaign or adjacent CUDA cells; reinforces host class or protocol but is not Alpha/Beta primary identity |
| `tooling_only` | Packaging, desktop engineering, or documentation process — not training-path acceptance |
| `neither` | Outside Alpha/Beta happy paths (e.g. MoE refuse, failed cohort without path claim) |

Also present as a **file** (not a directory): `2026-08-10-cuda-phase5-repeatability-retention.json` — retention/custody schedule for the successful Phase 5 anchor; treat as custody tooling for that packet (`tooling_only` companion).

| Directory | One-line claim boundary | Tag |
| --- | --- | --- |
| `docs/operations/evidence/2026-07-27-desktop-release/` | 10/10 clean local Mac packaging builds at commit `1038ecdd…`; ad-hoc signed review packages only — not public notarized release or training acceptance. | `tooling_only` |
| `docs/operations/evidence/2026-07-27-mlx-lm-acceptance/` | Historical two-run MLX-LM QLoRA `measured-run-pass` for same Qwen2.5-0.5B-4bit artifact under older plan/bundle contracts — not current v5/v3 fingerprint identity. | `supports_path_alpha` |
| `docs/operations/evidence/2026-07-28-qwen3-moe-admission/` | Exact Qwen3 30B-A3B MoE path: plan/compile/dependency/packed-checkpoint OK; **refused** at model-data on unified-memory shortfall — not training acceptance. | `neither` |
| `docs/operations/evidence/2026-07-29-documentation-drift-audit/` | Immutable point-in-time documentation drift audit (32 findings remediated in PR #14); no runtime training claim. | `tooling_only` |
| `docs/operations/evidence/2026-08-05-qwen2-mlx-lm-acceptance/` | Phase 6 baseline: two clean v5/v3 MLX QLoRA ladders to `measured-run-pass` for exact Qwen2.5 artifact at source `14ed44b5…` (bundle fingerprint `f1d175…`) — not quality/CUDA/general Qwen2. | `supports_path_alpha` |
| `docs/operations/evidence/2026-08-05-qwen2-mlx-lm-exact-source-refresh/` | **Path Alpha freeze primary:** two fresh clean ladders to `measured-run-pass` at source `71925515…`, fingerprint `ca2548cf…` — exact artifact/host/runtime/dataset/policy only. | `supports_path_alpha` |
| `docs/operations/evidence/2026-08-06-smollm2-cuda-lora-single-acceptance/` | **Path Beta freeze primary:** one five-job CUDA LoRA single ladder to `measured-run-pass` on Ubuntu/RTX 3050 for exact SmolLM2 revision — not repeatability, quality, or semantic adapter reload. | `supports_path_beta` |
| `docs/operations/evidence/2026-08-09-cuda-phase0-recovery-supplement/` | Recovery integrity for protected bytes bound to the Aug 6 acceptance packet — not new training, timing, or readiness claims. | `supports_path_beta` |
| `docs/operations/evidence/2026-08-09-cuda-phase5-repeatability-anchor/` | Phase 5 **failed** conditioning (telemetry capture-invalid); repeatability **not** established; five measured slots planned-not-started. | `neither` |
| `docs/operations/evidence/2026-08-10-cuda-phase5-repeatability-anchor/` | Freeze-listed related Beta support: five-slot SmolLM2-135M LoRA single BF16 repeatability on RTX 3050 at source `3bfec547…` — different commit/dataset from primary acceptance; not quality/release. | `supports_path_beta` |
| `docs/operations/evidence/2026-08-10-cuda-phase6-method-matrix/` | Historical Phase 6 method-matrix dispositions; no method met three-of-three promotion at that cohort — not Path Beta primary, not confirmatory stability. | `supporting_related` |
| `docs/operations/evidence/2026-08-10-cuda-phase6-remediation-matrix/` | Remediation cohort: Full exploratory promotion without confirmatory stability; no Phase 7 auth from that packet alone. | `supporting_related` |
| `docs/operations/evidence/2026-08-10-cuda-phase6-confirmatory-stability/` | Exact-host SmolLM2-135M **Full** five-slot confirmatory stability at source `2bc4d9a3…` — authorizes bounded Phase 7 procedure; not LoRA Path Beta identity. | `supporting_related` |
| `docs/operations/evidence/2026-08-10-cuda-phase7-scale-staircase/` | Stopped Phase 7 scale staircase: planner ledger + one 135M LoRA result + thermal/admission stop; immutable history, not stable Phase 7 cell. | `supporting_related` |
| `docs/operations/evidence/2026-08-11-cuda-phase7-same-family-stability/` | Three stable same-family Phase 7 cells (135M LoRA, 135M Full, 360M LoRA); 1.7B and 360M Full planned-not-started — not Path Beta primary freeze. | `supporting_related` |
| `docs/operations/evidence/2026-08-11-cuda-phase7-breadth-amendment/` | Pre-execution reviewed Qwen3-0.6B LoRA breadth contract only — no runtime training result. | `supporting_related` |
| `docs/operations/evidence/2026-08-11-cuda-phase7-breadth-parameter-correction/` | Parameter-semantics correction (unique loaded params vs serialized elements); first breadth cohort stopped at model-data — not stability. | `supporting_related` |
| `docs/operations/evidence/2026-08-11-cuda-phase7-breadth-stability/` | Stable exact-host Qwen3-0.6B LoRA Phase 7 breadth cell (three exploratory slots) — not Alpha/Beta primary identity. | `supporting_related` |
| `docs/operations/evidence/2026-08-11-cuda-phase8-guarded-frontier/` | Three guarded one-axis frontiers / Phase 9 candidate selection from bounded pilots only on Qwen3-0.6B LoRA — not full-train OOM campaign, not Beta primary. | `supporting_related` |
| `docs/operations/evidence/2026-08-11-cuda-phase9-endurance/` | Three 300-update endurance slots + eight job-control exercises pass on frozen Qwen3-0.6B LoRA host/source — no semantic CUDA reload, no quality. | `supporting_related` |
| `docs/operations/evidence/2026-08-11-cuda-phase10-certification/` | Campaign aggregation: 149 planned / 58 started / 47 native-pass+protocol-valid; certifies bounded RTX 3050 campaign cells — **not** Aptus 0.2 release readiness; does not redefine Path Beta primary acceptance. | `supporting_related` |

### Evidence tag counts (directories only)

| Tag | Count |
| --- | ---: |
| `supports_path_alpha` | 3 |
| `supports_path_beta` | 3 |
| `supporting_related` | 11 |
| `tooling_only` | 2 |
| `neither` | 2 |
| **Total directories** | **21** |

---

## Cross-links for later M1 work packages

| Follow-on | Path |
| --- | --- |
| Promise inventory / gap register | `M1-gap-register.csv` / `M1-promise-audit.md` (M1.1, M1.4, M1.5 — not this file) |
| Path Alpha re-proof + runbook | Mission M3 → `docs/guides/path-alpha-mlx-operator.md` |
| Path Beta re-proof + runbook | Mission M4 → `docs/guides/path-beta-cuda-lora-operator.md` |
| Freeze sources | `M0-PATH-ALPHA-FREEZE.md`, `M0-PATH-BETA-FREEZE.md` |

## Self-check

- [x] Journey A bound to Path Alpha freeze identity (MLX Qwen2.5 QLoRA).
- [x] Journey B bound to Path Beta freeze identity (CUDA SmolLM2 LoRA handoff).
- [x] Every step tagged `works | partial | missing | unknown`.
- [x] All 21 top-level evidence directories listed with claim boundary + tag.
- [x] No invented digests; hashes only where copied from freezes/evidence docs already cited.
- [x] Docs-only audit; no `src/` edits; no measured runs.
