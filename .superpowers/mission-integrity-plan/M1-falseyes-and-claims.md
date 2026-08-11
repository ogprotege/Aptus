# M1 — False-yes hunt and claim-language audit

> **Status:** Complete (read-only audit)  
> **Phase:** M1.4 + M1.5  
> **Date:** 2026-08-11  
> **Scope:** Code + docs + tests under `/Users/biscuit/Aptus`  
> **Non-claims:** No production behavior changed. No measured runtime pilot run. This is an integrity audit of fail-closed paths and public claim language, not a release gate.

---

## Summary of gaps

| Priority | Item | Finding |
| --- | --- | --- |
| **P2** | Full FP16 | Planner + plan-contract reject viable `full`+`fp16`; **no dedicated unittest** pins the refusal message or status (coverage is only indirect via capability docs / multi-device BF16 selection). |
| **P2** | CUDA profile on Mac (browser `aptus serve`) | Native macOS desktop handoff is hard-closed and tested. Browser workbench without `window.aptusDesktop` can still present local Run controls for a CUDA bundle; runtime eventually exits (`CUDA hardware is unavailable`), so this is a **UI soft false-yes risk**, not a silent measured pass. |
| **P2** | Claim tagline | README / mission hero “will actually run” is marketing-tightened by body text, but can be skim-read as stronger than evidence-scoped “eligible / conditional / pilot-required.” |
| — | Full FSDP, quantized FSDP, MLX Full, stale schema, MoE near-match, experimental methods | **Pass** (trustworthy no) with multi-layer code + tests + docs. |

**No P0 live false-yes found** on the audited public planning/compile/select/job paths for the eight named scenarios.  
**No P1 code-path false-yes found.** Residual items are test/UI/copy hygiene.

---

## Part A — False-yes hunt

Legend:

- **pass (trustworthy no)** — Aptus refuses, marks unsupported/nonselectable, or requires replan; multi-layer defense; tests or strong contract pins.
- **gap** — Missing defense, or defense exists but user-visible “yes” can still appear without clear refusal.
- **unknown** — Not established from this static audit.

### 1. Full FP16 request

| Field | Value |
| --- | --- |
| **Status** | **pass** (minor test-coverage gap → **P2**) |
| **Behavior** | Fail-closed: Full trains only when participating devices declare BF16; otherwise precision becomes `fp16` and Full is marked unsupported. Plan validation also rejects any viable `full`+`fp16` payload. |

**Code paths**

- `/Users/biscuit/Aptus/src/aptus/planning.py` (~653–662): precision is BF16 only if every participating device has `supports_bf16`; else `fp16`. Then:

  ```text
  if method == Method.FULL and precision == "fp16":
      unsupported.append(
          "Full-parameter FP16 training is fail-closed in Aptus v0.2 because the
           generated mixed-precision path does not retain verified FP32 trainable
           master weights."
      )
  ```

- `/Users/biscuit/Aptus/src/aptus/planning.py` (~400–410): single-device Full is “compatible” only when `device.supports_bf16` on CUDA (affects device selection; refusal still comes from precision rule).
- `/Users/biscuit/Aptus/src/aptus/plan_contract.py` (~2766–2773): rejects feasible/conditional candidates with `method=="full"` and `precision=="fp16"`.

**Docs**

- `docs/reference/capability-matrix.md` — “FP16 full | Always unsupported | No launch”
- `docs/guides/choose-a-method.md` — Full requires BF16; FP16 path rejected
- `docs/product/current-capabilities.md` — “Explicitly unsupported: Full-parameter FP16 training”
- `docs/reference/configuration-defaults.md` — CUDA full is BF16 only

**Tests**

- **No** dedicated test asserting Full FP16 → `UNSUPPORTED` with the fail-closed reason.
- Indirect: multi-device selection tests exercise `supports_bf16=False` for adapters (`tests/aptus/test_planning.py` device-selection cases), not Full-without-BF16 as the primary assertion.

**False-yes risk**

- Low for production paths (planner + contract). Residual risk is regression without a pin → **P2** add `test_full_fp16_is_fail_closed`.

---

### 2. Full FSDP

| Field | Value |
| --- | --- |
| **Status** | **pass** |
| **Behavior** | Explicit unsupported; remains visible in the matrix. |

**Code paths**

- `/Users/biscuit/Aptus/src/aptus/planning.py` (~605–608): Full + FSDP → unsupported with calibrated-transient reason.
- `/Users/biscuit/Aptus/src/aptus/methods/registry.py` (~60–61): Full `supported_distributions=("single", "ddp")` only — registry gate also refuses FSDP for Full.
- LoRA FSDP remains **conditional** (pilot-required), not Full.

**Tests**

- `tests/aptus/test_planning.py` — `test_full_fsdp_is_closed_and_lora_fsdp_requires_pilot`

**Docs**

- README support table; `current-capabilities.md` “Explicit unsupported records for full FSDP”; capability matrix.

---

### 3. Quantized FSDP

| Field | Value |
| --- | --- |
| **Status** | **pass** |
| **Behavior** | int8-LoRA + FSDP and QLoRA + FSDP are outside the verified matrix (unsupported). |

**Code paths**

- `/Users/biscuit/Aptus/src/aptus/planning.py` (~601–603):

  ```text
  if distribution == FSDP and method in {INT8_LORA, QLORA}:
      unsupported.append("... outside the verified v0.2 compiler matrix.")
  ```

- `/Users/biscuit/Aptus/src/aptus/plan_contract.py` (~3117–3120): rejects FSDP + `{int8-lora, qlora}` on plan validate.
- Registry: int8-LoRA and QLoRA CUDA bindings list `("single", "ddp")` only (no FSDP). QLoRA MLX is single-only.

**Tests**

- `tests/aptus/test_planning.py` — `test_quantized_fsdp_is_fail_closed`

**Docs**

- capability matrix, current-capabilities, README outside-coverage column.

---

### 4. MLX Full

| Field | Value |
| --- | --- |
| **Status** | **pass** |
| **Behavior** | Full has no MLX-LM runtime binding. On MPS hardware, Full is not viable; only LoRA/QLoRA MLX candidates can be conditional. |

**Code paths**

- `/Users/biscuit/Aptus/src/aptus/methods/registry.py` (~49–74): Full `supported_backends=("cuda",)` and only `transformers-peft-cuda` binding.
- `/Users/biscuit/Aptus/src/aptus/planning.py` (~443–478): on MPS, Full infers `TrainingRuntime.PYTORCH_MPS` (not MLX-LM); `registered_runtime_contract_for` returns no binding → unsupported “no registered full compiler on mps.”
- `/Users/biscuit/Aptus/src/aptus/plan_contract.py` (~3121–3123): MLX-LM methods limited to `{lora, qlora}`.
- Workbench: `RunStage.tsx` note — “This MLX path executes LoRA and QLoRA only. DoRA, full-parameter training, and resume are not supported.”

**Tests**

- `tests/aptus/test_planning.py` — `test_apple_unified_memory_yields_only_pilot_required_mlx_candidates` asserts viable methods == `{LORA, QLORA}` only.
- `tests/aptus/test_methods.py` — runtime bindings separate CUDA/MLX; Full not bound to MLX.
- `web/src/stages/RunStage.test.tsx` — MLX note includes full-parameter unsupported.

**Docs**

- README Apple Silicon outside coverage; current-capabilities “Full-parameter and DoRA training through MLX-LM.”

---

### 5. CUDA profile on Mac as if local CUDA train

| Field | Value |
| --- | --- |
| **Status** | **pass** for native desktop + local hardware probe; **gap (P2)** for browser workbench UX when not in desktop shell |
| **Behavior** | Local probe never invents CUDA on Darwin arm64. Manual CUDA facts describe a **remote** host. Desktop app hands CUDA off; does not start local CUDA jobs. Runtime probe fails closed if CUDA is missing. |

**Code / UI paths**

1. **Local inventory** — `/Users/biscuit/Aptus/src/aptus/profiling.py` (~926–958): if no CUDA devices and Darwin arm64 → one **MPS** unified-memory device; never pretends to be local CUDA. Empty device list → hard error: “CUDA hardware inspection is unavailable… Supply explicit manual facts.”
2. **Manual / CLI** — `--backend cuda` is allowed as **declared** target-host planning (README: “CUDA profiles describe a CUDA host; they do not enable CUDA work on the Mac.”).
3. **Desktop Run** — `web/src/stages/RunStage.tsx` (~56–60, 183–190): `desktopHandoff = macDesktop && !localAppleRuntime` → no Run radios / no “Start training”; copy: “The macOS app never submits CUDA work locally…”
4. **Execution fail-closed** — `src/aptus/execution.py` CUDA runtime probe exits with `CUDA hardware is unavailable` if `torch.cuda.is_available()` is false; device-index mismatch also exits.

**Tests**

- `tests/aptus/test_profiling.py` — Darwin arm64 probe → `Backend.MPS`, no bf16/4bit/8bit CUDA flags.
- `web/src/stages/RunStage.test.tsx` — “hands every desktop execution plan off to the CUDA host”; no Start training for non-MLX on mac desktop.

**Gap (P2)**

- Browser-only workbench (`aptus serve` without `window.aptusDesktop`): CUDA bundles can show local run controls. That is not a measured false-yes (job/runtime should fail without CUDA), but it is a **user-facing “yes path”** that the desktop shell intentionally closes. Consider aligning browser-on-Mac with handoff UX or an explicit “this host has no CUDA” pre-submit gate.

---

### 6. Stale plan schema load for compile/job

| Field | Value |
| --- | --- |
| **Status** | **pass** |
| **Behavior** | Pre-v6 / schema-less / stale-policy plans are not rehydrated for managed compile/job; HTTP **409 `replan_required`**; source bytes preserved. |

**Code paths**

- `SCHEMA_VERSION = "aptus.training-plan.v6"` — `src/aptus/domain.py`
- `UnsupportedPlanSchemaError` — domain + projects load
- `src/aptus/plan_contract.py` (~2148–2149): wrong schema → `replan_required` error text
- `src/aptus/api.py`: plan GET, compile, job submit map `UnsupportedPlanSchemaError` / `StaleModelPolicyError` → 409 `replan_required` with `required_schema` / `found_schema`
- Stale same-schema policy snapshot also replan_required (currency), not silent upgrade

**Tests**

- `tests/aptus/test_api.py` — `test_every_pre_v5_saved_plan_schema_requires_replanning_without_rewrite` (v4/v3/v2/None → 409, bytes unchanged)
- `test_stale_same_schema_policy_maps_to_replan_required`
- `test_stale_policy_job_submission_requires_replanning`
- `tests/aptus/test_plan_contract.py` — stale snapshot / stale policy / tampering distinctions
- Docs: current-capabilities strict replan for v4/v3/v2/schema-less and stale v5/v6 policy

**Note:** Test name says “pre-v5” while required schema is **v6**; still covers stale schema family. v5 stale-policy is covered by policy-digest tests.

---

### 7. MoE near-match without topology

| Field | Value |
| --- | --- |
| **Status** | **pass** |
| **Behavior** | Sparse identity markers without reviewed topology are **blocked** before dense-family recognition; near-matches yield no viable candidate / `NoFeasiblePlanError`. |

**Code paths**

- Model compatibility policy (Qwen3 MoE exact row only): sparse near-matches blocked; missing topology → `ModelPolicyDecisionKind.BLOCKED` with unreviewed-sparse reason.
- Planning applies `model_policy_rejection_reasons` to every candidate; near-match plans raise `NoFeasiblePlanError` when no feasible rows remain.
- Topology rail is presentation-only (current-capabilities): does not authorize memory subtraction or compile.

**Tests**

- `tests/aptus/test_model_compatibility.py` — `test_sparse_identity_markers_block_when_topology_is_missing` (Qwen2Moe / Mixtral markers without moe → BLOCKED)
- `tests/aptus/test_planning.py` — `test_qwen3_moe_near_match_has_no_viable_candidate` (architecture/layout/shared expert)
- `test_sparse_architecture_marker_without_topology_has_no_viable_candidate`

**Docs**

- current-capabilities sparse near-match / missing topology; ROADMAP first MoE slice exact-only.

---

### 8. Experimental method select attempt (DoRA etc.)

| Field | Value |
| --- | --- |
| **Status** | **pass** |
| **Behavior** | DoRA, BitFit, AdaLoRA, ShareLoRA experimental; LoReFT/AFLoRA/BiLoRA research-only; all `selectable=False`, no compiler/export. Planner enumerates only selectable methods. Selection API rejects non-feasible/nonselectable. |

**Code paths**

- `/Users/biscuit/Aptus/src/aptus/methods/registry.py` — DoRA et al. `selectable=False`, `compiler_id=None`, blocker text.
- `Method` enum (`domain.py`) is only `full|lora|int8-lora|qlora`.
- `plan_training` iterates `selectable_method_descriptors()` only.
- CLI `--prefer-method` choices = `Method` enum only (cannot prefer DoRA).
- API capabilities: full catalog for visibility; `methods` list = selectable IDs only.
- `select_candidate` — rejected if not feasible/conditional.
- `model_compatibility` — paths require selectable method.
- Client: `web/src/api.ts` normalizes method_catalog; non-selectable requires blocker.

**Tests**

- `tests/aptus/test_methods.py` — `test_researched_methods_do_not_become_selectable_by_presence`
- `tests/aptus/test_api.py` — method_catalog dora experimental + not selectable
- `tests/aptus/test_plan_contract.py` — `test_select_candidate_rejects_nonselectable_and_mutated_candidates`

**Docs**

- README outside coverage; current-capabilities 11-descriptor registry; NG-01 freeze (no DoRA compilers).

---

## Part A — status board

| # | Scenario | Status | Priority if gap |
| --- | --- | ---: | --- |
| 1 | Full FP16 request | pass (+ missing dedicated test) | P2 |
| 2 | Full FSDP | pass | — |
| 3 | Quantized FSDP | pass | — |
| 4 | MLX Full | pass | — |
| 5 | CUDA profile on Mac as local CUDA train | pass (desktop/probe); gap (browser UX) | P2 |
| 6 | Stale plan schema load for compile/job | pass | — |
| 7 | MoE near-match without topology | pass | — |
| 8 | Experimental method select (DoRA etc.) | pass | — |

---

## Part B — Claim-language audit

Policy source: `docs/product/claim-language.md` (normative). Forbidden class: “universally optimal,” “guaranteed to fit,” “perfect configuration,” “automatic best method,” “zero-risk training,” over-shortened CUDA certification, artifact-wide transfer of exact pilots.

### Surfaces scanned

| Surface | Path |
| --- | --- |
| README | `/Users/biscuit/Aptus/README.md` |
| Current capabilities | `/Users/biscuit/Aptus/docs/product/current-capabilities.md` |
| Claim language | `/Users/biscuit/Aptus/docs/product/claim-language.md` |
| CLI help sample | `/Users/biscuit/Aptus/src/aptus/cli.py` |
| Workbench sample | `web/src/stages/RunStage.tsx`, `web/src/demo.ts`, `web/src/components/FitLedger.tsx`, `web/src/components/CandidateComparison.tsx` |

### Findings

| Location | Phrase / pattern | Mark | Notes |
| --- | --- | --- | --- |
| `README.md` L7 | “Decide whether a fine-tune **will actually run** — before you spend the compute.” | **risk** (soft) | Product mission tagline. Body immediately scopes: not quality, not fit guarantee (L76–78). Claim-language policy prefers eligibility/conditional wording. Skim-risk only; not a hard overclaim if read with support table. |
| `README.md` L11 | “evidence ladder that **refuses unsupported claims**” | **ok** | Matches fail-closed stance. |
| `README.md` L75–78 | Recommendation = highest-ranked in catalog; **not** quality; **not** a guarantee unmeasured hardware will fit | **ok** | Explicit anti-overclaim. |
| `README.md` L155–178 | Spec-plan example statuses (infeasible/unsupported/conditional/feasible) + “declared examples” / gates still reject | **ok** | Separates real CLI math from declared model/hardware. |
| `README.md` L247–265 | Support table with “Outside current coverage” | **ok** | Full FSDP, quantized FSDP, MLX Full, DoRA, CUDA-on-macOS listed out. |
| `README.md` L240–243 | “CUDA profiles describe a CUDA host; they do not enable CUDA work on the Mac.” | **ok** | Aligns with scenario 5. |
| `README.md` L502–529 | Exact-commit / exact-host evidence boundaries; Phase 10 not release-ready | **ok** | Matches claim-language release section. |
| `docs/product/claim-language.md` entire | Normative allowed/forbidden lists | **ok** | Authority document; no drift found in the file itself. |
| `docs/product/current-capabilities.md` L4–92 | Long evidence preamble with exact IDs, non-transfer language | **ok** | Explicitly not model quality / broad CUDA / release readiness. |
| `docs/product/current-capabilities.md` L94–298 | “Available now” enumerated with pilot-required / conditional language | **ok** | Methods catalog, MoE exact-only, FSDP records. |
| `docs/product/current-capabilities.md` L343–363 | “Explicitly unsupported” list includes Full FP16, Full FSDP, quantized FSDP, CUDA on macOS, MLX Full | **ok** | Aligns with Part A. |
| `src/aptus/cli.py` L182 | `--backend` help: “Planned compute backend (default: cuda).” | **ok** | “Planned” not “local.” Mild default bias toward CUDA is intentional for remote planning; not “supports all.” |
| `src/aptus/cli.py` L262 | prefer-method: “cannot override feasibility” | **ok** | |
| `src/aptus/cli.py` L289 | packing: “unsupported and fail-closed” | **ok** | |
| `src/aptus/cli.py` L478 | hardware: “Inspect local CUDA hardware or fail-closed Apple Silicon inventory.” | **ok** | |
| `web/src/stages/RunStage.tsx` L180 | MLX: DoRA, full-parameter, resume not supported | **ok** | |
| `web/src/stages/RunStage.tsx` L186–188 | CUDA desktop handoff; never submits CUDA locally | **ok** | |
| `web/src/components/FitLedger.tsx` L52–53 | “Example values. No hardware inspection ran.” | **ok** | |
| `web/src/components/CandidateComparison.tsx` L124 | “Recommended” label | **ok** | UI label; server/docs define ranking within set (not “optimal”). |
| `web/src/demo.ts` | CUDA example demo plan | **ok** | Demo/example provenance badges used in UI. |
| Forbidden strings scan (`guaranteed`, `optimal`, `supports all`, `zero-risk`, `universally`) on README + `docs/product/*` | Only appear as **prohibited** examples in claim-language / mission plan, or in “not a guarantee” negation | **ok** | No live product claim uses forbidden positive form. |

### Claim-language summary

| Mark | Count | Disposition |
| --- | --- | --- |
| **ok** | majority of scanned surfaces | Keep |
| **drift** | 0 hard drifts | — |
| **risk** | 1 soft (README/mission hero “will actually run”) | Optional copy tighten in M2 claim hygiene; not a false-yes code path |

Legacy audit trees under `docs/audits/aptus-legacy/` and `Reference/` still contain historical “optimal” language about **pre-Aptus** systems; out of product surface scope for this audit (do not treat as current product claims).

---

## Recommended follow-ups (not done in M1)

1. **P2** — Add `test_full_fp16_is_fail_closed` in `tests/aptus/test_planning.py` (and optional plan_contract forge test).  
2. **P2** — Browser workbench: if host inventory is MPS-only, surface CUDA target-host handoff (or disable local Run) even without desktop bridge.  
3. **Copy** — Consider mission tagline footnote: “within declared facts and the evidence ladder,” already true in body.  
4. Carry residuals into M1 gap register / M2 refusal-integrity only if product prioritizes; **no P0 false-yes blocks M1 exit for this package**.

---

## Explicit non-claims of this audit

- Did not execute live CUDA or MLX pilots.  
- Did not re-run the full unittest/vitest suite in this subagent session (citations are from static code/test source review).  
- Did not audit every CLI subcommand help string line-by-line beyond plan/hardware/prefer-method samples.  
- Did not claim release readiness.
