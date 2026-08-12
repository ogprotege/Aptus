# M5 — Correction object specification (freeze)

> **Status:** APPROVED 2026-08-12 (owner: M5.1 approved — implement M5.2)  
> **Path:** Server-owned structured correction on plan outcomes  
> **Authority:** Mission integrity plan §12  
> **Implementation:** M5.2 in progress on branch `feat/mission-m5-correction`

## 1. Goal (one sentence)

After every plan attempt, the solo operator sees **one primary next action** —
select the recommended candidate, replan with concrete fact changes, or accept
that no supported path exists — with reasons derived only from planner data.

## 2. Non-goals

- Hyperparameter search / Optuna / continuous replan loops  
- Inventing methods, distributions, or runtimes outside the current catalog  
- Claiming optimality (“best possible”) or model quality  
- Replacing `rejection_reasons` free text or candidate digests  
- LLM-authored gates (optional later paraphrase only of this structure)

## 3. Situations → product behavior

| Situation | Detection | `kind` | Primary action |
| --- | --- | --- | --- |
| ≥1 **feasible** candidate | Plan succeeds; recommended.status = `feasible` | `select-candidate` | Use recommended (or pick another viable) |
| 0 feasible, ≥1 **conditional** | Plan succeeds; recommended.status = `conditional` | `select-candidate` | Use recommended; pilot required before full train |
| 0 viable (no feasible, no conditional) | `NoFeasiblePlanError` / HTTP 422 `no_feasible_plan` | `no-path` | Change facts listed in hints; do not invent methods |

**Note:** Current planner treats both `feasible` and `conditional` as
`candidate.feasible == true` and always returns a `TrainingPlan` when any
viable row exists. `NoFeasiblePlanResponse` covers the empty-viable case only.

`replan-with-fact-hints` is **not** a separate planner outcome. It is a
**presentation mode** of `no-path` (and optionally of conditional memory
warnings) when actionable fact hints exist. Wire kinds stay the three enum
values below.

## 4. Schema (normative)

Attach a single object. Field names use snake_case on the wire (Python/API).

### 4.1 `correction` object

```json
{
  "schema_version": "aptus.plan-correction.v1",
  "kind": "select-candidate | no-path",
  "summary": "string",
  "primary_reason_codes": ["string"],
  "recommended_candidate_id": "cand_… | null",
  "recommended_status": "feasible | conditional | null",
  "pilot_required": true,
  "ranking_objective": "quality | memory | speed | null",
  "fact_hints": [
    {
      "fact": "target.sequence_length",
      "direction": "decrease | increase | set | review",
      "why": "string",
      "source_reason_codes": ["infeasible_memory"]
    }
  ],
  "disallowed_suggestions": [
    {
      "code": "no_fsdp",
      "message": "Do not enable full FSDP; unsupported in v0.2."
    }
  ],
  "operator_next_step": {
    "action": "compile-recommended | confirm-pilot-then-train | change-facts",
    "label": "string"
  }
}
```

### 4.2 Field rules

| Field | Rules |
| --- | --- |
| `schema_version` | Constant `aptus.plan-correction.v1` |
| `kind` | `select-candidate` iff a recommended viable candidate exists; else `no-path` |
| `summary` | ≤ 240 chars; one sentence; no quality claims |
| `primary_reason_codes` | From `aptus.refusal` reason codes (stable). For select-candidate: ranking codes or empty/`pilot_required`. For no-path: top codes from rejected candidates (max 5, frequency then catalog order) |
| `recommended_candidate_id` | Required non-null for `select-candidate`; **null** for `no-path` |
| `recommended_status` | Status of recommended; null for no-path |
| `pilot_required` | true when recommended is conditional OR any viable path is pilot-required; true for no-path when only conditional rows were almost-viable (optional); default false for pure feasible recommend |
| `ranking_objective` | Copy of `target.objective` when plan exists; null on no-path if target not on error payload (use request target if available in API layer) |
| `fact_hints` | Derived only from refusal `changeable_facts` + mapped direction; max 5; **empty** when none actionable |
| `disallowed_suggestions` | Closed contracts that must not be proposed as “fixes”; always include catalog non-goals relevant to rejected rows (FSDP, multi-GPU without devices, packing, etc.) |
| `operator_next_step` | Single CTA for UI/CLI |

### 4.3 Direction mapping (deterministic)

From refusal `changeable_facts` tokens (substring match):

| Fact token contains | `direction` |
| --- | --- |
| `sequence_length` | `decrease` when memory/context overflow codes; else `review` |
| `effective_batch` / `micro_batch` | `decrease` for memory codes; else `review` |
| `vram` / `free memory` / `host_ram` / `disk` | `increase` |
| `reserve` | `decrease` (operator may lower reserve only if safe) |
| `method` / `distribution` / `training_runtime` / `backend` | `review` |
| `packing` | `set` (set packing=false) |
| default | `review` |

`why` is the refusal title + short explanation (from `RefusalGuidance`), not free LLM text.

### 4.4 Attachment points

| Surface | Attachment |
| --- | --- |
| `TrainingPlan` / plan JSON v6 | Top-level `correction` **presentation field** |
| `TrainingPlanResponse` (API) | Top-level `correction` |
| `NoFeasiblePlanResponse` | Top-level `correction` |
| Plan identity / digests | **Must not** include `correction` in plan_id / candidate_id material (same rule as refusal stderr: presentation only) |

**Identity rule (critical):** Building `correction` is pure function of plan (or no-feasible payload) + target objective. It must **not** alter:

- `plan_id`, candidate digests, policy snapshot bindings  
- `recommendation_rationale` text used in identity (leave existing rationale as-is)  

If plan identity hashing walks all top-level keys today, implement `correction` either:

1. **Preferred:** compute at serialization/API/CLI/UI boundary without storing on the domain `TrainingPlan` dataclass used for `plan_id_for_payload`, **or**  
2. Explicitly exclude `correction` from identity payload (document + test).

Recommended implementation path: `build_plan_correction(plan | no_feasible_payload) -> Correction` in `src/aptus/correction.py`, called from:

- API response builders  
- CLI after plan / no-feasible  
- Web client can recompute from candidates for offline parity **or** trust server field (prefer server field as source of truth when present)

## 5. Operator next-step mapping

| Situation | `operator_next_step.action` | Example `label` |
| --- | --- | --- |
| select-candidate + feasible | `compile-recommended` | “Compile recommended bundle” |
| select-candidate + conditional | `confirm-pilot-then-train` | “Run pilot, then confirm full train” |
| no-path + actionable hints | `change-facts` | “Change facts and replan” |
| no-path + no actionable hints | `change-facts` | “No supported path — review catalog refusals” |

UI: **one** primary button bound to that action. Secondary: expand full candidate table (already exists). No second navigation maze.

## 6. Examples (Alpha / Beta flavored)

### 6.1 Path Alpha–style success (conditional MLX QLoRA)

```json
{
  "schema_version": "aptus.plan-correction.v1",
  "kind": "select-candidate",
  "summary": "Use QLoRA single on MLX-LM; pilot is required before full train.",
  "primary_reason_codes": ["conditional_pilot_required"],
  "recommended_candidate_id": "cand_bec6f029a7417259d49c",
  "recommended_status": "conditional",
  "pilot_required": true,
  "ranking_objective": "memory",
  "fact_hints": [],
  "disallowed_suggestions": [
    {"code": "no_mlx_full", "message": "Do not switch to full fine-tuning on MLX; no full compiler is registered."},
    {"code": "no_multi_gpu", "message": "Do not enable DDP/FSDP on a single-device inventory."}
  ],
  "operator_next_step": {
    "action": "confirm-pilot-then-train",
    "label": "Run pilot, then confirm full train"
  }
}
```

### 6.2 Path Beta–style success (feasible CUDA LoRA)

```json
{
  "schema_version": "aptus.plan-correction.v1",
  "kind": "select-candidate",
  "summary": "Use LoRA single BF16 on CUDA; ranked under the speed objective among viable candidates.",
  "primary_reason_codes": [],
  "recommended_candidate_id": "cand_2fe2c0a05360293358f6",
  "recommended_status": "feasible",
  "pilot_required": true,
  "ranking_objective": "speed",
  "fact_hints": [],
  "disallowed_suggestions": [
    {"code": "no_fsdp", "message": "Do not enable full FSDP; unsupported in v0.2."}
  ],
  "operator_next_step": {
    "action": "compile-recommended",
    "label": "Compile recommended bundle"
  }
}
```

Note: even “feasible” rows still require measured preflight + pilot before full
train per runtime contracts; `pilot_required` stays honest.

### 6.3 No-path (sequence too long / memory)

```json
{
  "schema_version": "aptus.plan-correction.v1",
  "kind": "no-path",
  "summary": "No supported training path fits these facts; reduce sequence length or free more memory.",
  "primary_reason_codes": ["infeasible_memory", "sequence_length_exceeds_context"],
  "recommended_candidate_id": null,
  "recommended_status": null,
  "pilot_required": false,
  "ranking_objective": "memory",
  "fact_hints": [
    {
      "fact": "target.sequence_length",
      "direction": "decrease",
      "why": "Estimated memory exceeds usable device capacity / sequence exceeds context.",
      "source_reason_codes": ["infeasible_memory", "sequence_length_exceeds_context"]
    },
    {
      "fact": "target.effective_batch_size",
      "direction": "decrease",
      "why": "Estimated memory exceeds usable device capacity.",
      "source_reason_codes": ["infeasible_memory"]
    }
  ],
  "disallowed_suggestions": [
    {"code": "no_new_method", "message": "Do not invent a training method outside Full/LoRA/int8-LoRA/QLoRA."},
    {"code": "no_fsdp", "message": "Do not enable FSDP as a workaround; unsupported in v0.2."}
  ],
  "operator_next_step": {
    "action": "change-facts",
    "label": "Change facts and replan"
  }
}
```

## 7. Claim language (docs + UI copy)

Allowed:

- “Recommended within the enumerated candidate set under the stated objective.”  
- “No supported path for these facts.”  
- “Pilot required before full train.”  

Forbidden:

- “Optimal”, “best model”, “guaranteed to fit”, “will pass measured-run”.  
- Suggesting multi-GPU, FSDP, packing, or unregistered methods as the fix.

## 8. Implementation work packages (post-approval)

| ID | Work | Done when |
| --- | --- | --- |
| M5.2 | `src/aptus/correction.py` + unit tests (feasible / conditional / no-path / no unsupported suggestion) | Tests fail then pass |
| M5.2b | Wire into API `TrainingPlanResponse` + `NoFeasiblePlanResponse`; OpenAPI regenerate; client contracts | `generate_openapi --check` green |
| M5.2c | Web types + normalizer; Compare panel | Vitest + manual glance |
| M5.3 | One CTA button on Compare | Matches `operator_next_step` |
| M5.4 | CLI prints correction block (stdout or stderr section; JSON file stays plan-primary) | Spec-plan shows block |
| M5.5 | Guide “When Aptus refuses / corrects” | Doc tests pass |
| M5.6 | Intentional bad sequence/batch walk (CLI) for Alpha-like and Beta-like facts | Recorded in completion note |

## 9. Test matrix (minimum)

1. Plan with ≥1 feasible → `kind=select-candidate`, non-null recommended id, no invented methods in disallowed/hints  
2. Plan with only conditional recommended → `pilot_required=true`, CTA confirm-pilot  
3. `NoFeasiblePlanError` → `kind=no-path`, null recommended, fact_hints ⊆ refusal changeable facts  
4. Hints never include “enable FSDP” / “add DDP” when multi_gpu unsupported  
5. Plan identity unchanged when correction is attached at presentation boundary  

## 10. Owner approval

| Decision | Owner |
| --- | --- |
| Schema `aptus.plan-correction.v1` as above | ☑ Approve |
| Identity exclusion (presentation-only) | ☑ Approve |
| Three situations mapping | ☑ Approve |
| Proceed to M5.2 implementation | ☑ Yes |

**Approval recorded:** 2026-08-12 owner chat: “M5.1 approved — implement M5.2”

---

## Appendix A — Reuse of M2 refusal

M5 **reuses** `aptus.refusal.RefusalGuidance` / `web/src/lib/refusal.ts` as the
code catalog. M5 does **not** replace per-candidate refusal blocks; it adds one
plan-level correction summary so operators are not left with only a wall of
rejected rows.
