# M2 Refusal / Rejection Reason Catalog

| Metadata | Value |
| --- | --- |
| **Status** | Active M2 inventory |
| **Date** | 2026-08-11 |
| **Authority** | Mission integrity plan M2 — mine code/docs/UI for operator-facing “why no” surfaces; invent stable `reason_code` identities where the product currently emits free-text only |
| **Scope** | Planning candidate matrix, model policy, plan contract / replan, compile, validate findings, run/API lifecycle errors, CLI/UI presentation |
| **Sources** | `docs/reference/error-codes.md`, `src/aptus/planning.py`, `src/aptus/model_compatibility.py`, `src/aptus/plan_contract.py`, `src/aptus/api.py`, `src/aptus/methods/registry.py`, `docs/guides/troubleshooting.md`, `web/src/stages/CompareStage.tsx`, `web/src/components/CandidateComparison.tsx`, `web/src/components/StatusBadge.tsx`, `src/aptus/cli.py` |
| **Related mission gaps** | P-03 (explicit refuse), P-18 (changeable facts), P-12 (multi-GPU dual vocab), P-01 (reject visibility) |

## How to read this catalog

- **reason_code** — stable snake identity invented for M2. Product code today mostly stores **free-text** strings on `CandidatePlan.rejection_reasons` or API `error` / finding `code` fields. Map free-text with the **example free-text** column.
- **surface** — primary emission surface (`plan.candidate`, `plan.api`, `policy`, `plan_contract`, `method_catalog`, `compile`, `validate`, `run.api`, `cli`).
- **Candidate status ladder** (planner only): `unsupported` (capability/policy/matrix) → `infeasible` (arithmetic/resource) → `conditional` (viable but pilot-required / envelope warning) → `feasible`.
- **changeable_facts** — operator-editable planning/runtime fact fields that can clear the refusal without a product release; `none-in-catalog` when only a product change, recompile path, or host process action applies.
- **operator_actionable** — `yes` if an operator can act with current facts/tools; `partial` if action is replan/recompile/wait; `no` if product matrix only.

---

## 1. Planner candidate refusals (`rejection_reasons`)

Emitted by `_estimate_candidate_with_policy` in `src/aptus/planning.py`. Concatenated as `unsupported + infeasible + conditional`. Status is first non-empty class. Viable candidates may still carry **conditional** reason strings while `feasible=true`.

| reason_code | surface | user_visible_title | explanation | changeable_facts | operator_actionable | existing_tests | docs | example free-text / pattern |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `full_fp16` | plan.candidate | Full FP16 closed | Full-parameter training requires BF16 devices; FP16 path lacks verified FP32 master weights. | `hardware.devices[].supports_bf16` (use BF16 GPU); or choose non-`full` method | yes | plan_contract rejects full FP16 execution; **no dedicated named unittest pin for FP16 free-text** (M1 residual) | troubleshooting (BF16); method-registry full gates; capability-matrix | `Full-parameter FP16 training is fail-closed in Aptus v0.2 because the generated mixed-precision path does not retain verified FP32 trainable master weights.` |
| `full_fsdp` | plan.candidate | Full FSDP closed | Full-parameter FSDP fails closed (FP32 upcast / full-state export uncalibrated). | `distribution` (use single/DDP when multi-GPU); or method ≠ full | yes | `test_full_fsdp_is_closed_and_lora_fsdp_requires_pilot` | troubleshooting “Full FSDP is rejected” | `Full-parameter FSDP is fail-closed in Aptus v0.2 because the pinned runtime upcasts trainable shards and full-state export to FP32…` |
| `quantized_fsdp` | plan.candidate | Quantized FSDP closed | `int8-lora` / `qlora` + FSDP outside verified v0.2 matrix. | `distribution` (single/DDP); or method LoRA/full (LoRA FSDP only) | yes | `test_quantized_fsdp_is_fail_closed` | troubleshooting Full FSDP section | `{int8-lora\|qlora} with FSDP is outside the verified v0.2 compiler matrix.` |
| `mlx_full` | plan.candidate | MLX full not compiled | On MPS, full resolves to unavailable runtime binding (no MLX/PyTorch-MPS full compiler). | `method` (lora/qlora on MLX); or CUDA host for full | yes | `test_apple_unified_memory_yields_only_pilot_required_mlx_candidates` (matrix shape); methods registry (full CUDA-only) | method-registry; capability-matrix | `{pytorch-mps\|mlx-lm} has no registered full compiler on mps.` (pattern: `{runtime} has no registered {method} compiler on {backend}.`) |
| `multi_gpu_on_single` | plan.candidate | Multi-GPU needs ≥2 devices | DDP/FSDP enumerated on single-GPU inventory stay visible as **unsupported**. | `hardware.devices` (add GPUs); or accept single distribution | yes | `test_single_gpu_keeps_multi_gpu_strategies_visible_but_unsupported` | capability-matrix multi-GPU open; NG-07 | `{ddp\|fsdp} requires at least two GPUs.` |
| `registry_distribution_unsupported` | plan.candidate | Method distribution unsupported | Method descriptor does not list the distribution. | `distribution` / method choice within registry | yes | `test_registry_distribution_support_is_an_authoritative_gate` | method-registry | `The {method_id} registry contract does not support {distribution} distribution.` |
| `runtime_compiler_missing` | plan.candidate | No runtime compiler | Resolved training runtime has no binding for method×backend (includes MLX full, CUDA method on MPS, etc.). | `target.training_runtime`, `method`, host/backend | yes | planning + methods tests | method-registry runtime_bindings | `{runtime} has no registered {method} compiler on {backend}.` |
| `runtime_distribution_unsupported` | plan.candidate | Compiler distribution unsupported | Binding exists but not for this distribution (e.g. MLX LoRA/QLoRA single only). | `distribution` → single on MLX | yes | MoE / MLX planning tests | method-registry MLX bindings | `The {runtime} {method_id} compiler does not support {distribution} distribution.` |
| `no_compute_device` | plan.candidate | No compute device | Hardware inventory empty. | `hardware.devices` (rescan / supply devices) | yes | planning fixtures assume devices | troubleshooting hardware scan | `At least one supported compute device is required.` |
| `mixed_compute_backends` | plan.candidate | Mixed backends | Participating devices span multiple backends. | `hardware.devices` (homogeneous set) | yes | none named | — | `A candidate cannot mix compute backends.` |
| `runtime_backend_mismatch` | plan.candidate | Runtime/backend mismatch | Training runtime requires a compute backend the devices do not provide. | `target.training_runtime`, devices | yes | none named | — | `{runtime} requires {backend} compute.` |
| `sequence_length_exceeds_context` | plan.candidate | Sequence > context | Target sequence longer than model context. | `target.sequence_length`, `model.context_length` | yes | none named (fact validation may also catch) | — | `Requested sequence length exceeds the model context length.` |
| `unsupported_dataset_schema` | plan.candidate | Dataset schema unsupported | Schema not in allowed set. | `dataset.schema_name` ∈ text, prompt-completion, instruction-output, messages, mixed | yes | generation schema tests related | dataset-schemas | `Unsupported dataset schema: {name}.` |
| `task_not_sft` | plan.candidate | Non-SFT closed | Only supervised fine-tuning compiled. | `target.task` = `sft` | yes | `test_non_sft_and_packing_fail_closed` | — | `Aptus v0.2 compiles supervised fine-tuning (task='sft') only.` |
| `packing_unsupported` | plan.candidate | Packing closed | Sequence packing not implemented. | `target.packing` = false | yes | `test_non_sft_and_packing_fail_closed` | CLI help packing | `Sequence packing is not implemented in the v0.2 masking compiler; set packing=false.` |
| `max_wall_time_closed` | plan.candidate | Wall-time limit closed | Process manager does not enforce graceful deadline. | `target.max_wall_time_minutes` = null / omit | yes | none named | — | `max_wall_time_minutes is fail-closed…` |
| `qlora_four_bit_device` | plan.candidate | QLoRA needs 4-bit device | CUDA QLoRA requires `supports_4bit` on every participating device (MLX uses model metadata instead). | devices with 4-bit; or MLX four-bit model; or other method | yes | capability / planning paths | troubleshooting | `QLoRA requires explicit runtime-native four-bit support on every participating device.` |
| `int8_lora_device` | plan.candidate | INT8 LoRA needs 8-bit | Every participating GPU must support 8-bit. | devices `supports_8bit`; or other method | yes | none named | — | `Eight-bit LoRA requires explicit eight-bit support on every participating GPU.` |
| `unsupported_model_family` | plan.candidate | Unknown adapter family | Target modules catalog miss for non-full methods. | `model.family` ∈ catalog (llama, mistral, gemma, qwen, qwen3_moe) | yes | catalog tests; planning | method-registry LoRA families | `Unsupported model family '{family}'. Supported families: …` |
| `batch_not_divisible_world` | plan.candidate | Batch not divisible by world size | Global batch % world_size ≠ 0. | `target.effective_batch_size`, GPU count / distribution | yes | none named | — | `Global batch {n} is not divisible by world size {w}.` |
| `explicit_batch_arithmetic` | plan.candidate | Explicit batch mismatch | micro × accum × world ≠ effective. | `target.micro_batch_size`, `gradient_accumulation_steps`, `effective_batch_size` | yes | none named | — | `Explicit micro-batch, accumulation, and world-size arithmetic does not equal the requested effective batch.` |
| `exact_batch_unpreserved` | plan.candidate | Exact batch not preserved | Selected micro/accum could not preserve requested global batch. | batch fields / world size | yes | none named | — | `Exact global batch arithmetic could not preserve the requested batch.` |
| `infeasible_memory` | plan.candidate | Point estimate over memory | Even point estimate exceeds usable per-device memory (after reserve). | `hardware` capacity/reserve; `target` batch/seq; smaller method/quant; model size | yes | memory/planning tests; API no_feasible fixtures use related wording | troubleshooting “No feasible candidate”, preflight OOM | `Even the point estimate exceeds usable per-device memory.` |
| `conditional_upper_envelope` | plan.candidate | Upper envelope over memory | Point fits; uncalibrated upper exceeds usable memory → **conditional**. | same as memory; still pilot-required | partial | memory bound naming tests | methodology memory-estimation | `Point estimate fits, but the uncalibrated heuristic upper envelope exceeds usable per-device memory.` |
| `host_ram_infeasible` | plan.candidate | Host RAM short | CUDA host staging heuristic exceeds free/total host RAM. | host RAM; method; world size (DDP/FSDP copies) | yes | none named | troubleshooting | `Host RAM is below the minimum model-loading heuristic.` |
| `disk_infeasible` | plan.candidate | Disk short | Free disk below staging/pilot/checkpoint estimate. | free disk; model/dataset sizes; retention assumptions | yes | none named | — | `Free disk is below the compiled staging, bounded-pilot, and three-checkpoint retention estimate.` |
| `conditional_fsdp_pilot` | plan.candidate | FSDP pilot required | LoRA FSDP viable only as **conditional** (uncalibrated sharding). | none-in-catalog for product calibration; operator can still pilot | partial | `test_full_fsdp_is_closed_and_lora_fsdp_requires_pilot` | troubleshooting LoRA FSDP conditional | `FSDP uses a simplified uncalibrated per-device sharding prior; the exact wrapping and transient path requires a real-model pilot.` |
| `conditional_pilot_required` | plan.candidate | MLX pilot required | Every non-unsupported MLX candidate is conditional (provisional unified-memory). | none-in-catalog for estimate guarantee; free memory / smaller candidate for later gates | partial | `test_apple_unified_memory_yields_only_pilot_required_mlx_candidates` | troubleshooting MLX sections | `MLX-LM support is pilot-required: the unified-memory estimate is provisional and cannot guarantee that the exact model and data fit.` |
| `policy_path_rejection` | plan.candidate | Model policy path miss | Policy decision does not match this method/distribution/modules/runtime path. | method/distribution/runtime; or model facts for path-matched policies | yes | MoE path tests; model_compatibility tests | model-policy-snapshot; current-capabilities | See §2 path reasons (e.g. Qwen3 MoE only single MLX-LM QLoRA attention adapters). |
| `policy_blocked_near_match` | plan.candidate | Recognized but blocked config | BLOCKED decision: identity near-match fails constraints + path reason. | model quantization/topology/layers/identity facts | yes | `test_qwen3_moe_near_match_has_no_viable_candidate` | Qwen3 MoE admission evidence | `decision.reason` + `path_rejection_reason` (see §2). |
| `moe_near_match` | plan.candidate | MoE near-match blocked | Alias for Qwen3 MoE blocked / non-path matrix rows (mission-required row). | MoE topology, 4-bit layout, no shared expert; only MLX QLoRA single attention path executable | yes | `test_qwen3_moe_near_match_has_no_viable_candidate`; `test_qwen3_moe_allows_only_attention_only_single_mlx_qlora` | troubleshooting MLX model-data; evidence 2026-07-28 | `The exact Qwen3 MoE identity was recognized, but this revision does not match…` **and/or** `Qwen3 MoE is executable only as single-device MLX-LM QLoRA with attention-only adapters.` |

### Conditional vs rejection note

Conditional candidates remain **selectable / recommended-eligible**. Their “reasons” are **warnings**, not hard fails. UI lists them under the same `rejection_reasons` field (naming debt for M2 styling).

---

## 2. Model policy block / path reasons

Emitted via `model_policy_rejection_reasons` and inspection/API compatibility responses (`src/aptus/model_compatibility.py`). Domain also has structured `ModelPolicyReasonCode` (kebab) on decisions/bindings — **not** copied into planner free-text today.

| reason_code | surface | user_visible_title | explanation | changeable_facts | operator_actionable | existing_tests | docs | example free-text / codes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `policy_unsupported` | policy + plan.candidate | No / blocked policy | Catch-all for non-matching sparse/unknown/blocked policy (mission-required). | model identity, quantization, topology; or stay on dense family path | yes | `tests/aptus/test_model_compatibility.py`; planning MoE | model-policy docs; error-codes POLICY_SNAPSHOT_* | See rows below; API decision `kind` ∈ blocked / unknown |
| `qwen3_moe_path_only` | policy | MoE executable path only | Only single-device MLX-LM QLoRA attention-qkvo. | method/runtime/distribution/target modules | yes | MoE planning tests | current-capabilities | `Qwen3 MoE is executable only as single-device MLX-LM QLoRA with attention-only adapters.` |
| `qwen3_moe_identity` | policy | MoE identity mismatch | Requires exact qwen3_moe / Qwen3MoeForCausalLM. | model architecture / model_type / family | yes | model_compatibility | — | `MoE execution requires the exact reviewed qwen3_moe and Qwen3MoeForCausalLM provider identity.` · code `identity-mismatch` |
| `qwen3_moe_layout` | policy | MoE quantization layout | 4-bit default + 8-bit router-gate overrides required. | `quantization_layout`, bits | yes | model_compatibility | — | `Qwen3 MoE execution requires the exact four-bit plus eight-bit router-gate MLX quantization layout.` · `quantization-layout-mismatch` |
| `qwen3_moe_topology` | policy | MoE topology incomplete | Full provider expert topology required. | moe expert facts | yes | — | — | `Qwen3 MoE execution requires the complete provider-declared expert topology.` · `topology-incomplete` |
| `qwen3_moe_shared_expert` | policy | Shared expert unsupported | First MoE contract forbids shared expert. | topology shared-expert flags | yes | — | — | `The first Qwen3 MoE MLX-LM contract does not support a shared expert.` · `shared-expert-unsupported` |
| `qwen3_moe_four_bit` | policy | MoE four-bit required | Explicit four-bit metadata required. | `quantization_bits` = 4 | yes | — | — | `The first Qwen3 MoE MLX-LM contract requires explicit four-bit model metadata.` · `four-bit-required` |
| `qwen3_moe_blocked_inspection` | policy | MoE recognized, config blocked | Inspection blocked composite reason. | layout/topology as above | yes | near-match tests | admission evidence | `The exact Qwen3 MoE identity was recognized, but this revision does not match the reviewed four-bit default…` |
| `qwen2_path_only` | policy | Qwen2 path only | Reviewed footprint: single MLX-LM QLoRA dense causal modules. | method/runtime/distribution | yes | Qwen2 policy tests | Path Alpha freeze | `The reviewed Qwen2 runtime footprint is executable only as single-device MLX-LM QLoRA with dense q/k/v/o/gate/up/down adapters.` |
| `qwen2_identity` | policy | Qwen2 identity | qwen / qwen2 / Qwen2ForCausalLM. | model identity | yes | — | — | `Dense Qwen2 execution requires the reviewed qwen, qwen2, and Qwen2ForCausalLM…` |
| `qwen2_layers` | policy | Qwen2 layer count | Exactly 24 layers. | `model.layers` | yes | — | — | `…requires exactly 24 transformer layers.` · `layer-count-mismatch` |
| `qwen2_layout` | policy | Qwen2 uniform 4-bit GS64 | Uniform four-bit group-size-64, no overrides. | quantization layout | yes | — | — | `…uniform four-bit, group-size-64…` |
| `qwen2_dense` | policy | Qwen2 dense required | No MoE on this footprint. | remove moe config | yes | — | — | `…requires dense topology with no MoE configuration.` · `dense-topology-required` |
| `qwen2_four_bit` | policy | Qwen2 four-bit required | Explicit four-bit metadata. | `quantization_bits` | yes | — | — | `…requires explicit four-bit model metadata.` |
| `qwen2_blocked_inspection` | policy | Qwen2 recognized, config blocked | Composite blocked inspection. | layers/layout | yes | — | — | `The Qwen2 identity was recognized, but this configuration does not match…` |
| `unreviewed_sparse_model` | policy | Unreviewed sparse | Sparse models need exact reviewed policy. | use reviewed sparse artifact; or dense model | yes | planning sparse markers | — | `Sparse model execution requires an exact reviewed model compatibility policy.` · `unreviewed-sparse-model` |
| `unknown_policy` | policy | No family policy | No exact model-type/architecture policy (planner may still allow dense via family catalog). | model_type/architecture; or proceed under family-recognized | partial | — | — | `No exact Aptus model-family compatibility policy matches…` · `no-policy-match` |
| `invalid_compatibility_facts` | policy | Malformed compatibility facts | Contradictory/malformed model compatibility facts. | correct inspection/attested facts | yes | receipt rejection tests | — | `Model compatibility facts are malformed or contradictory.` · `invalid-compatibility-facts` |
| `family_recognized` | policy | Family only (not a hard refuse) | Dense family map; planner still decides. | n/a (informational) | n/a | ModelPolicyPanel family-only | UI ModelPolicyPanel | `The provider identity maps to an existing dense Aptus family…` · `family-recognized` |

---

## 3. Plan-level API / CLI outcomes

| reason_code | surface | user_visible_title | explanation | changeable_facts | operator_actionable | existing_tests | docs | example free-text / envelope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `no_feasible_plan` | plan.api / cli | No feasible plan | All 12 method×distribution candidates rejected (no feasible/conditional). HTTP **422**. Body includes full candidate matrix + policy chain. | any facts that clear at least one candidate (§1–2) | yes | `test_api.py` no_feasible_plan suite; `test_planning` no_feasible provider | error-codes; troubleshooting “No feasible candidate” | API `error: no_feasible_plan`; message `No feasible or conditional training plan: {deduped reasons}` |
| `stale_plan_replan` | plan_contract + plan.api | Replan required | Saved plan schema outdated **or** coherent v5/v6 plan’s decision/snapshot no longer current on host. HTTP **409** `replan_required`. Malformed/tampered is **invalid**, not replan. | none-in-catalog for old bytes; re-plan from preserved source facts | partial | `test_stale_same_schema_policy_maps_to_replan_required`; `test_plan_contract` StaleModelPolicyError; bootstrap replan_required | error-codes replan_required; troubleshooting replan section | Message: `This saved plan predates the current executable contract…` **or** `The saved plan uses policy semantics that are no longer current; replan_required.` |
| `candidate_selection_rejected` | plan.api | Candidate selection rejected | Explicit select of stale/unknown/mutated/rejected/already-selected candidate. HTTP **409**. | choose viable different `candidate_id` on current plan | yes | select_candidate tests; API 409 | error-codes | `Candidate is stale, unknown…` / `mutated…` / `rejected or nonselectable` / `already selected…` |
| `experimental_method` | method_catalog | Experimental / research-only not selectable | Descriptor exists with `selectable=false` and `blocker`; **not** in 12-row matrix. | none-in-catalog until product ships compiler | no | `test_researched_methods_do_not_become_selectable_by_presence`; API method catalog | method-registry lifecycle | Blockers e.g. DoRA: `No Aptus compiler, calibrated estimator, or verified export/reload contract exists yet.` |

---

## 4. Compile refusals

| reason_code | surface | user_visible_title | explanation | changeable_facts | operator_actionable | existing_tests | docs | example free-text / code |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `compile_output_not_empty` | compile / api | Bundle path conflict | No-clobber: non-empty output refused. | new empty output path | yes | generation FileExistsError cases | error-codes `path_conflict`; compile guide | `Bundle output is not empty: {path}` → API `path_conflict` |
| `compile_plan_invalid` | compile | Plan failed current contract | `validate_plan_payload` / policy currency before compile. | replan under current schema | partial | generation policy snapshot binding tests | error-codes | CLI: `Persisted plan failed the current executable contract…` |
| `compile_mlx_dataset_row` | compile | MLX dataset row refused | MLX collator refuses text/content-only rows. | dataset schema/rows | yes | generation MLX dataset tests | dataset-schemas | `MLX-LM compilation refuses text rows…` / `content-only rows…` |
| `compile_static_validation_failed` | compile | Generated bundle invalid | Post-generate static validation failed. | facts / product bug fix; recompile empty path | partial | generation static tests | validation-states | `Generated bundle failed static validation.` |
| `archive_integrity` | compile | Archive construction refused | Symlinks, unstable inventory, non-canonical zip. | clean tree; no symlinks | yes | archive reject tests | — | various `Bundle archive…` ValueErrors |

---

## 5. Validate / findings (host)

Normative codes in `docs/reference/error-codes.md`. Representative high-impact rows (not every MANIFEST_* duplicate):

| reason_code | surface | user_visible_title | explanation | changeable_facts | operator_actionable | existing_tests | docs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `PLAN_CONTRACT_ERROR` | validate | Plan contract error | Schema/identity/semantic plan rule failed. | recompile from trusted plan | partial | validation/plan_contract tests | error-codes |
| `PLANNER_PARITY_MISMATCH` | validate | Planner parity mismatch | Replan from bound facts ≠ stored candidates/recommendation. | recompile; do not hand-edit plan | partial | validation parity | error-codes |
| `POLICY_SNAPSHOT_*` | validate | Policy snapshot integrity/currency | Missing/JSON/contract/noncanonical/digest/path. Host digest vs current registry → replan currency. | recompile; replan if host currency | partial | validation policy tests | error-codes; troubleshooting static validation |
| `MANIFEST_*` | validate | Manifest integrity | Schema, digests, missing/mismatched files. | recompile empty path | yes | generation/validation | error-codes |
| `RUNTIME_VALIDATION_FAILED` | validate | Runtime validation failed | Portable validate.py nonzero. | env/model/data per log | yes | package-free tests | validation-states |
| `PREFLIGHT_METRICS_*` / `PILOT_METRICS_*` | validate | Preflight/pilot metrics invalid or unbound | Measured evidence missing/misbound. | rerun ordered gate; fix environment | yes | execution/validation | error-codes |
| `mlx_model_data_memory_refuse` | run/validate | MLX refuses before load | Unified-memory admission fail-closed with required/available/shortfall. | free memory; smaller plan | yes | generation/execution MLX; historical Qwen3 admission | troubleshooting “MLX model-data refuses before the model loads” |

---

## 6. Run / job / API lifecycle refusals

| reason_code | surface | user_visible_title | explanation | changeable_facts | operator_actionable | existing_tests | docs | example free-text / envelope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `job_prerequisite_not_met` | run.api | Gate order skipped | Managed action before required validation state. HTTP **409**. | complete prior validation stage | yes | `test_job_prerequisite_failure_is_typed_as_http_409` | error-codes; run-states | `Cannot start {action}: validation-report.json has state … required …` + structured action/required_state/current_state/reason |
| `active_job_conflict` | run.api | Active job conflict | Another Aptus accelerator action holds lease. | wait/cancel owner job | yes | execution cancel/lease tests | troubleshooting active job | `active_job_conflict` |
| `runtime_unavailable` | run.api | Runtime interpreter missing | No measured/configured Python for bundle. | configure runtime endpoint / install deps | yes | runtime API tests | troubleshooting hardware / doctor | `runtime_unavailable` |
| `runtime_configuration_invalid` | run.api | Runtime probe failed | Selected Python failed training-runtime probe. | fix interpreter packages | yes | — | — | `runtime_configuration_invalid` |
| `desktop_execution_disabled` | run.api | Desktop execution off | Service started with local execution disabled. | enable runtime or transfer bundle | yes | — | error-codes | `desktop_execution_disabled` |
| `desktop_session_required` | run.api | Desktop session cookie | Private macOS service missing session cookie. | use app-launched session | yes | — | error-codes | `desktop_session_required` |
| `mlx_resume_unsupported` | run (bundle) | MLX resume closed | `--resume-from` always rejected. | none-in-catalog; start uninterrupted new run | no (product boundary) | `test_generated_mlx_resume_arguments_fail_closed` | troubleshooting resume; NG-05 | `MLX-LM resume is unsupported…` / `--resume-from is unsupported for MLX-LM…` |
| `train_admission_blocked` | run.api | Train after pilot rejected | Deep admission recheck failed (capacity, env, policy currency, pilot binding). | fix capacity/env; replan if currency | yes | execution pilot authorization | troubleshooting “Train submission is rejected after pilot” | job/authorization error strings; may map to `replan_required` |
| `project_*_mismatch` | plan.api | Project binding mismatch | Plan/bundle not on revision, snapshot mismatch, revision conflict. | use matching project revision / recompile | yes | projects/API tests | error-codes | `project_plan_mismatch`, `project_bundle_mismatch`, etc. |
| `path_forbidden` / `path_not_found` / `path_conflict` | api | Path errors | FS permission, missing path, no-clobber conflict. | correct paths / permissions / destinations | yes | API path tests | error-codes | details string from OS |
| `request_validation` | api | Request shape invalid | Pydantic strict rejection. HTTP **422**. | fix request fields | yes | OpenAPI client contracts | error-codes | Pydantic `details` |
| `invalid_request` | api | Contract value error | Generic ValueError → 400. | correct value/operation | yes | many | error-codes | `details: str(error)` |

---

## 7. Minimum coverage checklist (mission-required codes)

| Required identity | Catalog row(s) | Status |
| --- | --- | --- |
| `full_fp16` | §1 | covered (free-text mapped; test pin gap) |
| `full_fsdp` | §1 | covered + tests + docs |
| `quantized_fsdp` | §1 | covered + tests + docs |
| `mlx_full` | §1 | covered via runtime_compiler_missing pattern |
| `multi_gpu_on_single` | §1 | covered + tests |
| `stale_plan_replan` | §3 | covered + tests + docs |
| `moe_near_match` | §1–2 | covered + tests + admission evidence |
| `experimental_method` | §3 | covered (catalog, not candidate matrix) |
| `infeasible_memory` | §1 | covered |
| `conditional_pilot_required` | §1 | covered (MLX + FSDP conditional siblings) |
| `policy_unsupported` | §2 | covered (umbrella + specific MoE/Qwen2) |
| `no_feasible_plan` | §3 | covered + API 422 matrix |

---

## 8. UI / CLI gaps (reasons without structured changeable_facts)

Where the operator **sees** a refuse but **does not** get structured “change these facts” guidance today:

| Surface | What shows | Gap |
| --- | --- | --- |
| **Compare stage** (`web/src/stages/CompareStage.tsx`) | Inspected candidate: free-text `rejection_reasons` as plain warning list; StatusBadge only (feasible/infeasible/unsupported/conditional). No mapping to fact fields, reason_code, or suggested alternate candidate. | Highest P-18 gap |
| **Candidate comparison table** (`CandidateComparison.tsx`) | Fit badge only; **does not** show rejection reason text in the table/cards — operator must inspect | Reasons hidden until inspect |
| **StatusBadge** | Tone by status string; no reason taxonomy | Dual-vocab risk for multi-GPU **unsupported** vs “ready” (P-12) if styling confuses conditional/feasible |
| **No-viable panel** | Generic “Change the target or hardware facts” | No per-reason change map |
| **no_feasible_plan client normalize** (`web/src/api.ts`) | Generic rationale: “Review the rejection reasons before changing facts.” Candidates retained | No structured changeable_facts |
| **ModelPolicyPanel** | Structured badges (blocked / family only / path matched) + reason prose | Better than candidate reasons, still no fact-field checklist |
| **CLI `spec-plan` / `plan`** | Full plan JSON via `_write_json` / `to_primitive` — includes `rejection_reasons` arrays | Machine-readable free-text only; no reason_code, no CLI human summary of “why / change what” |
| **CLI no_feasible** | Exception path (NoFeasiblePlanError) depending on command wiring; API returns 422 matrix | Operator must parse JSON candidates |
| **Troubleshooting guide** | Generic “No feasible candidate” causes (BF16, VRAM, host RAM, distribution, model fact) | Not keyed by reason_code or free-text fingerprint |
| **error-codes.md** | Stable for API errors and validation findings; **not** for planner free-text | Planner catalog missing from normative inventory |
| **Conditional reasons in `rejection_reasons`** | Conditional pilot/envelope strings share the rejection list | Naming confuses hard refuse vs pilot-required viable row |
| **Experimental methods** | Method registry/API blockers only | Workbench Compare never lists them as refused rows — invisible “no” unless browsing method catalog |

### Presentation rules for later M2 implementation (inventory only)

1. Always show **status + free-text + invented reason_code** for unsupported/infeasible.
2. For conditional, label **pilot-required / envelope warning**, not “rejected.”
3. Attach **changeable_facts** from this catalog when rendering.
4. Keep all 12 matrix rows visible (never hide multi-GPU/experimental-adjacent fails).
5. Do not style planner-supported multi-GPU as measured-ready (P-12).

---

## 9. Synonym / dedupe notes

| Collapsed into | Synonyms / near-duplicates avoided |
| --- | --- |
| `infeasible_memory` | “point estimate exceeds available memory”, “exceeds usable per-device VRAM” (older fixtures) — same class as current “usable per-device memory” |
| `stale_plan_replan` | schema predate message; policy semantics current message; bootstrap `replan_required` object; API `error: replan_required` |
| `policy_unsupported` | umbrella over blocked/unknown/path miss; prefer specific MoE/Qwen2 codes when free-text matches |
| `conditional_pilot_required` | MLX pilot string; FSDP pilot kept as `conditional_fsdp_pilot` sibling |
| `runtime_compiler_missing` | covers `mlx_full` pattern; keep `mlx_full` as mission alias |
| `multi_gpu_on_single` | distinct from `registry_distribution_unsupported` and `runtime_distribution_unsupported` |

---

## 10. Source index (absolute paths)

| Role | Path |
| --- | --- |
| Planner reasons | `/Users/biscuit/Aptus/src/aptus/planning.py` |
| Policy free-text + codes | `/Users/biscuit/Aptus/src/aptus/model_compatibility.py`, `/Users/biscuit/Aptus/src/aptus/domain.py` (`ModelPolicyReasonCode`) |
| Replan / stale | `/Users/biscuit/Aptus/src/aptus/plan_contract.py` (`StaleModelPolicyError`) |
| API envelopes | `/Users/biscuit/Aptus/src/aptus/api.py`, `/Users/biscuit/Aptus/docs/reference/error-codes.md` |
| Method blockers | `/Users/biscuit/Aptus/src/aptus/methods/registry.py` |
| Compile | `/Users/biscuit/Aptus/src/aptus/generation.py` |
| Jobs | `/Users/biscuit/Aptus/src/aptus/execution.py` |
| UI | `/Users/biscuit/Aptus/web/src/stages/CompareStage.tsx`, `.../components/CandidateComparison.tsx`, `StatusBadge.tsx`, `ModelPolicyPanel.tsx` |
| CLI | `/Users/biscuit/Aptus/src/aptus/cli.py` |
| Operator docs | `/Users/biscuit/Aptus/docs/guides/troubleshooting.md` |
| Mission gap owners | `/Users/biscuit/Aptus/.superpowers/mission-integrity-plan/M1-gap-register.csv` (P-03, P-18) |

---

*End of M2 refusal catalog inventory. Implementation of structured reason_code emission and UI changeable_facts is out of scope for this file.*
