# Remainder Completion Implementation Plan

> **Status:** Active | **Authority:** Implementation plan (subordinate to claim language, the remainder freeze, and the Lane 7 spec) | **Applies to:** Aptus remainder completion after R0 | **Audience:** Agents executing hygiene and Lane 7 after owner go, then Lanes 8–10 one at a time | **Last reviewed:** 2026-08-31 | **Review by:** After Lane 7 starts, or when this remainder is superseded

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement **one approved lane** task-by-task. Do not start Lane 7 code until the owner approves `docs/superpowers/specs/2026-08-31-lane-7-gemma4-unified-design.md` and says go. Do not start Lanes 8–10 from this file alone. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the named Gemma 4 remainder with fail-closed identities, so Aptus stops returning a false `no-policy-match` for types it already named, without growing 0.2 or claiming quality.

**Architecture:** R0 (merged, PR #109) named the program. Lane 7 is a second exact identity under family `gemma4`. Lane 8 is a census slice only if a pin omits `v_proj` on k-equals-v layers. Lane 9 is a new MoE topology. Lane 10 is a new dataset-facts surface. The 0.2 referee stays frozen.

**Tech Stack:** Existing Aptus Python core, MLX-LM bundles, claim-language docs. No OpenAPI hand-edits. No `aptus-work/` commits. No mlx-lm fork in these increments.

## Global Constraints

- Remainder freeze: `docs/superpowers/specs/2026-08-26-remainder-program-design.md`
- Lane 7 spec: `docs/superpowers/specs/2026-08-31-lane-7-gemma4-unified-design.md` (PROPOSED until owner go)
- Not M10. Not 0.3. Not a 0.2 ship. Do not grow 0.2.
- One increment at a time. Spec → owner "go" → code + tests + docs → PR.
- Always open a GitHub PR after pushing. Merge only when every named check is SUCCESS. Use `gh pr merge --merge`. Fast-forward local `main` afterward.
- Do not retrain Journey A, B, B2, E2B, or E4B. Do not overwrite those trees.
- Do not commit `aptus-work/` or `web/node_modules`.
- Do not force Gemma 4 31B or Qwen3 30B-A3B trains on this Mac.
- Do not start a Qwen3 MoE train inside a Gemma 4 increment.
- `measured-run-pass` is not quality. Gold is not quality. Recitation is not Use.
- Never hand-edit `docs/reference/openapi.v1.json` or `web/src/generated/openapi.ts`.
- Never alias `gemma4_unified_text` onto `model.gemma4.mlx.v1`.

---

## Who owns what

This is the Veriformis-style contract for the remainder. Aptus still does
not ingest PDFs. Veriformis still does not train.

| Concern | Owner | Aptus will not |
| --- | --- | --- |
| Dense Gemma 4 identity (`gemma4_text` / `Gemma4ForConditionalGeneration`) | Lane 6, already on main | Alias unified or MoE onto it |
| Unified 12B identity | Lane 7 | Pretend the Hub type is unknown; pretend mlx-lm loaded it if it did not |
| k-equals-v omitted-`v_proj` census | Lane 8, only if a pin needs it | Weaken llama/qwen/mistral instance totals |
| 26B-A4B expert topology | Lane 9 | Call resident weight "active parameters"; train Qwen3 MoE here |
| Image/audio JSONL | Lane 10 | Grow plan statuses or a 0.3 referee |
| mlx-lm architecture loader | mlx-lm | Vendor a fork to make 12B look like `gemma4.py` |
| Envelope (RAM/disk) | This Mac's ledger | Force a 31B or 30B-A3B train |
| Quality / specialist / Use | Operator + later eval | Treat gold, loss, or recitation as a product yes |
| 0.2 referee | Cut note | Add stages, statuses, or rank-formula churn |

## Three identities, three jobs

Do not concatenate these. Do not let one policy row cover all three.

| Identity | Job | Aptus command surface |
| --- | --- | --- |
| Dense Gemma 4 | Language-tower text SFT on `Gemma4ForConditionalGeneration` | Existing inspect / Compare / compile / emit-run |
| Unified Gemma 4 | Language-tower text SFT on `Gemma4UnifiedForConditionalGeneration` | Lane 7 policy, then emit-run in a **new** work dir |
| Gemma 4 MoE | Sparse 26B-A4B with real expert integers | Lane 9 only; Compare stays an honest no until that row exists |

Vision is not a fourth model identity. It is a **dataset-facts** path on an
already-admitted pin (Lane 10). Text SFT on the language tower must keep
working while vision is refused or separately gated.

## Split ownership (inspect vs compile vs run vs claim)

1. **Inspect** maps provider type + architecture to an Aptus family and
   emits a compatibility decision. `no-policy-match` means Aptus does not
   have a row. A typed `unsupported by the current compiler contract`
   means Aptus **does** have a row and the bound runtime cannot execute it.
2. **Compare / compile** may emit a bundle only for a matched path. They
   do not prove the host can train it.
3. **emit-run / ladder** on this Mac produce envelope refuse or
   `measured-run-pass`. Parent owns completion.
4. **Claim language** may follow only the strongest of those four. A
   policy row is not "Aptus supports Gemma 4."

Do not ask inspect to honor LM Studio folders. Do not ask Veriformis
`evaluation.jsonl` to become the MLX valid tail. Recitation gold is not
that holdout. See `docs/guides/aptus-veriformis-handoff.md`.

## Working order

```text
R0 freeze                         merged PR #109 (2026-08-31)
  → Hygiene                       current-capabilities honesty (Lane 6 leftover)
  → Lane 7 spec                   this PR, PROPOSED
  → Lane 7 code                   only after owner go
  → Lane 8                        only if the Lane 7 pin omits v_proj
  → Lane 9                        26B-A4B MoE
  → Lane 10                       vision / multimodal SFT
  → stop, or owner names a new program
```

---

## Hygiene (Lane 6 leftover, same change as the first Lane 7 docs PR or a tiny preceding docs PR)

**Files:**

- Modify: `docs/product/current-capabilities.md`
- Modify: `docs/reference/capability-matrix.md` (CUDA "Gemma remains license-excluded" sentence — keep CUDA Gemma 2/3 exclusion; do not let it erase Lane 6)
- Modify: `docs/product/claim-language.md` only if a new allowed sentence is required ("second exact identity under family gemma4")

**Do**

1. State that dense Gemma 4 MLX LoRA/QLoRA is a named Lane 6 family path,
   conditional, pilot-required, not CUDA, not quality, not Path Alpha.
2. Keep CUDA Gemma 2/3 license-exclusion as CUDA-campaign history.
3. Do not claim unified 12B, MoE, or vision here until those lanes close.

**Done when.** An operator reading current capabilities can tell dense
Gemma 4 MLX exists and CUDA Gemma exclusion is a different sentence.
No new method. No 0.2 growth.

- [ ] Owner includes this hygiene in the first Lane 7-adjacent docs change
- [ ] Wording reviewed against claim-language.md
- [ ] PR opened and merged with full-green CI

---

## Lane 7: Gemma 4 unified (12B)

Do not start code until the owner approves the Lane 7 spec and says go.

**Files:**

- Modify: `src/aptus/inspection.py` (`_PROVIDER_MODEL_TYPE_ALIASES`, architecture guard)
- Modify: `src/aptus/model_compatibility.py` (new policy constants, `_GEMMA4_UNIFIED_POLICY`, append to `MODEL_COMPATIBILITY_POLICIES`)
- Modify: `src/aptus/evidence.py` (`policy.gemma4-unified.mlx.v1`)
- Modify: `src/aptus/plan_contract.py` only if path/evidence tables need the new IDs; do not put unified onto `model.gemma4.mlx.v1`
- Modify: `tests/aptus/test_inspection.py`
- Modify: `tests/aptus/test_model_compatibility.py`
- Modify: `tests/aptus/test_policy_snapshot.py`
- Modify: `docs/reference/model-policy-snapshot.md`
- Modify: `docs/reference/api.md`
- Modify: `docs/reference/evidence-records.md`
- Modify: `docs/guides/inspect-results.md`
- Modify: `docs/product/current-capabilities.md` (hygiene + this identity)
- Modify: `CHANGELOG.md`
- Test: `tests/aptus/test_documentation.py` only if Markdown counts change in a docs-only follow-up; behavior PR updates the same docs

**Interfaces:**

- Consumes: Lane 6 `model.gemma4.mlx.v1` exact identity
  (`gemma4` / `gemma4_text` / `Gemma4ForConditionalGeneration`)
- Produces: `model.gemma4-unified.mlx.v1` exact identity
  (`gemma4` / `gemma4_unified_text` /
  `Gemma4UnifiedForConditionalGeneration`); inspect family `gemma4` with
  raw `model_type` preserved

### Task 1: Failing inspect mapping test

- [ ] **Step 1: Write the failing test**

```python
def test_gemma4_unified_text_maps_to_gemma4_family_not_dense_policy(self) -> None:
    commit = "a" * 40
    transport = SequenceTransport(
        [
            FakeResponse(
                {
                    "model_type": "gemma4_unified_text",
                    "architectures": [
                        "Gemma4UnifiedForConditionalGeneration"
                    ],
                    "quantization": {
                        "group_size": 64,
                        "bits": 4,
                        "mode": "affine",
                    },
                    "hidden_size": 3840,
                    "intermediate_size": 15360,
                    "num_hidden_layers": 48,
                    "max_position_embeddings": 262144,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 8,
                    "vocab_size": 262144,
                    "enable_moe_block": False,
                    "num_experts": None,
                },
                {"X-Repo-Commit": commit},
            ),
            FakeResponse(
                {"cardData": {"license": "apache-2.0"}},
                {"X-Repo-Commit": commit},
            ),
        ]
    )
    result = inspect_huggingface_model(
        "mlx-community/gemma-4-12b-it-4bit",
        "73bcf09092aa277861d5a191b989b666f7f32e8f",
        transport=transport,
    )
    self.assertEqual(result["status"], "ok")
    self.assertEqual(result["facts"]["family"], "gemma4")
    self.assertEqual(result["facts"]["model_type"], "gemma4_unified_text")
    self.assertEqual(
        result["facts"]["architecture"],
        "Gemma4UnifiedForConditionalGeneration",
    )
    self.assertEqual(result["facts"]["layers"], 48)
    self.assertIsNone(result["facts"]["moe"])
    self.assertNotEqual(
        result["compatibility"].get("reason"),
        "No exact Aptus model-family compatibility policy matches this "
        "provider model type and architecture.",
    )
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m unittest \
  tests.aptus.test_inspection.InspectionTests.test_gemma4_unified_text_maps_to_gemma4_family_not_dense_policy -v
```

Expected: FAIL on family `gemma4_unified_text` or on leftover
`no-policy-match`.

- [ ] **Step 3: Add the architecture-guarded alias**

In `src/aptus/inspection.py`:

```python
_PROVIDER_MODEL_TYPE_ALIASES = {
    # existing keys unchanged
    "gemma4_text": "gemma4",
    "gemma4_unified_text": "gemma4",
}
_UNIFIED_GEMMA4_ARCHITECTURES = {
    "Gemma4UnifiedForConditionalGeneration",
}
```

In `_catalog_family`, after the Gemma 3 architecture guard, add:

```python
if (
    normalized == "gemma4_unified_text"
    and architecture not in _UNIFIED_GEMMA4_ARCHITECTURES
):
    return raw_model_type, (
        "Provider model_type 'gemma4_unified_text' was not mapped to catalog "
        f"family 'gemma4' because architecture {architecture!r} is not an "
        "explicitly supported Gemma 4 unified architecture."
    )
```

Do not map `gemma4_unified_text` onto dense
`Gemma4ForConditionalGeneration`.

- [ ] **Step 4: Re-run the inspect test**

Same command as Step 2. It may still fail at compatibility until Task 2
adds the policy. Family and architecture asserts must already pass.

### Task 2: Failing compatibility test, then the second policy

- [ ] **Step 1: Write the failing tests**

```python
def test_gemma4_unified_does_not_match_dense_policy(self) -> None:
    subject = ModelCompatibilitySubject(
        family="gemma4",
        model_type="gemma4_unified_text",
        architecture="Gemma4UnifiedForConditionalGeneration",
        layers=48,
        quantization_bits=4,
        quantization_layout=QuantizationLayout(4, 64),
        moe=None,
    )
    decision = evaluate_model_compatibility(subject)
    self.assertNotEqual(decision.policy_id, GEMMA4_POLICY_ID)


def test_gemma4_unified_family_matches_qlora_and_lora_paths(self) -> None:
    subject = ModelCompatibilitySubject(
        family="gemma4",
        model_type="gemma4_unified_text",
        architecture="Gemma4UnifiedForConditionalGeneration",
        layers=48,
        quantization_bits=4,
        quantization_layout=QuantizationLayout(4, 64),
        moe=None,
    )
    decision = evaluate_model_compatibility(subject)
    response = compatibility_response_v1(decision)
    self.assertEqual(decision.kind, ModelPolicyDecisionKind.PATH_MATCHED)
    self.assertEqual(decision.policy_id, "model.gemma4-unified.mlx.v1")
    self.assertEqual(
        {path.path_id for path in decision.paths},
        {
            "mlx-lm.qlora.single.gemma4-unified.v1",
            "mlx-lm.lora.single.gemma4-unified.v1",
        },
    )
    self.assertEqual(response["status"], "conditional")
    self.assertEqual(response["supported_runtime"], "mlx-lm")


def test_dense_gemma4_still_matches_lane_6_policy(self) -> None:
    subject = ModelCompatibilitySubject(
        family="gemma4",
        model_type="gemma4_text",
        architecture="Gemma4ForConditionalGeneration",
        layers=35,
        quantization_bits=4,
        quantization_layout=QuantizationLayout(4, 64),
        moe=None,
    )
    decision = evaluate_model_compatibility(subject)
    self.assertEqual(decision.policy_id, GEMMA4_POLICY_ID)
```

- [ ] **Step 2: Run them and confirm unified matching fails**

```bash
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m unittest \
  tests.aptus.test_model_compatibility.ModelCompatibilityPolicyTests.test_gemma4_unified_does_not_match_dense_policy \
  tests.aptus.test_model_compatibility.ModelCompatibilityPolicyTests.test_gemma4_unified_family_matches_qlora_and_lora_paths \
  tests.aptus.test_model_compatibility.ModelCompatibilityPolicyTests.test_dense_gemma4_still_matches_lane_6_policy -v
```

- [ ] **Step 3: Add `model.gemma4-unified.mlx.v1`**

Mirror `_GEMMA4_POLICY` with a new exact_identity on
`gemma4_unified_text` / `Gemma4UnifiedForConditionalGeneration`. Unique
`policy_id` and unique path IDs. Same dense `moe is null` constraint.
Same bits 1–16 QLoRA / unquantized LoRA split. Append to
`MODEL_COMPATIBILITY_POLICIES`. Add `policy.gemma4-unified.mlx.v1` in
`evidence.py`.

If bound mlx-lm **cannot** load the architecture, do not ship
`PATH_MATCHED` / `conditional`. Ship identity recognition plus a typed
compiler-contract unsupported (Lane 7 spec Exit B). The tests above then
assert that status and reason instead of `PATH_MATCHED`. Decide Exit A vs
B from the host probe in Task 3 **before** choosing which assertion is
normative. Do not mark conditional on a loader Aptus does not have.

- [ ] **Step 4: Re-run the compatibility tests and the dense Gemma 4 suite**

```bash
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m unittest tests.aptus.test_model_compatibility tests.aptus.test_inspection tests.aptus.test_policy_snapshot -q
```

Expected: OK.

### Task 3: Host probe (this Mac, uncommitted)

- [ ] **Step 1: Re-inspect the pin into a new directory**

```bash
# new dir; do not overwrite E2B/E4B
.venv/bin/aptus inspect \
  --model mlx-community/gemma-4-12b-it-4bit \
  --revision 73bcf09092aa277861d5a191b989b666f7f32e8f
```

Record architecture, layers, bits, `attention_k_eq_v`,
`num_kv_shared_layers`, moe-null, and compatibility. Keep JSON under
`aptus-work/` (uncommitted).

- [ ] **Step 2: Probe whether bound mlx-lm can load that architecture**

Do not alias the config to `gemma4` / `Gemma4ForConditionalGeneration`.
If load fails, Lane 7 is Exit B: typed compiler-contract unsupported,
docs, tests, PR. That is a valid close.

- [ ] **Step 3: Lane 8 gate**

From inspect + loaded module census:

- k-count equals v-count → skip Lane 8 for this pin
- omitted `v_proj` on k-equals-v, `k_proj` present ≥ 1 → finish Lane 8
  before any train
- any other asymmetry → refuse

- [ ] **Step 4: If Exit A, emit-run into a new work dir**

Same magisterium recipe shape as E2B only as a **starting** compare. New
plan, new bundle path. Envelope refuse is success. `measured-run-pass` is
success. Neither is quality.

### Task 4: Docs, changelog, PR

- [ ] **Step 1: Update the reference and product pages listed under Files**
- [ ] **Step 2: Run the repository-wide checks in CONTRIBUTING.md**
- [ ] **Step 3: Commit only Aptus source and docs. Never `aptus-work/`.**
- [ ] **Step 4: Push, open the PR, wait for every named check SUCCESS, `gh pr merge --merge`, fast-forward local `main`.**

**Lane 7 done when.** Spec approved + owner go + inspect not
`no-policy-match` + Exit A or Exit B + dense path unchanged + PR merged.

- [ ] Lane 7 spec approved
- [ ] Owner "go"
- [ ] Inspect recorded
- [ ] Second exact identity (no alias)
- [ ] Compile / emit-run **or** typed compiler refuse
- [ ] Measured ladder or envelope refuse (Exit A only)
- [ ] PR opened and merged

---

## Lane 8: k-equals-v omitted-`v_proj`

Skip as a standalone PR if Lane 7 never needs it.

**Do**

1. Persist inspect-time per-target counts.
2. Allow omitted `v_proj` only for family `gemma4`, only when the layer
   contract is k-equals-v, and only when `k_proj` still appears at least
   once. mlx-lm: `use_k_eq_v = attention_k_eq_v and not is_sliding`.
3. Parent verification still matches the bound census. No any-subset
   backslide for llama, qwen, or mistral.

**Files (when needed):**

- Modify: `src/aptus/plan_contract.py` (`mlx_trainable_target_instance_total`)
- Modify: `tests/aptus/test_plan_contract.py`
- Modify: `docs/guides/inspect-results.md`
- Modify: generated-bundle parent-verification tests if the census flows into `train.py`

**Done when.** Tests refuse asymmetric k/v, accept documented k-equals-v
omit-v, and leave other families strict. No 31B train is required.

- [ ] Needed by the Lane 7 pin, or deferred until a later pin needs it
- [ ] Spec (or Lane 7 spec section) approved
- [ ] Census tests
- [ ] PR opened and merged

Do not write Lane 8 code from this file until that gate is recorded.

---

## Lane 9: Gemma 4 26B-A4B MoE

Do not start until Lane 7 is closed and a Lane 9 spec is approved.

**Who owns what**

| Concern | Owner | Aptus will not |
| --- | --- | --- |
| Exact architecture + immutable revision | Lane 9 spec | Size-tweak the dense policy |
| Total vs active parameters | Sparse estimator | Substitute active for resident |
| Expert / router / adapter census | Compiler | Train a zero-trainable or accidental-dense graph |
| Qwen3 30B-A3B | Already has a compiler row; envelope already failed | Start that train inside this increment |

**Do**

1. Exact architecture plus immutable revision. Real expert integers stay
   MoE facts. Null experts stay dense.
2. Separate total versus active parameters.
3. Expert / router / adapter-target census.
4. Sparse memory/disk estimator that does not substitute active for
   resident.
5. One MLX QLoRA path, pilot-required. Compare stays an honest no until
   the row exists.
6. Measured ladder only if this Mac admits it. If not, conditional plus
   envelope refuse is success.

**Done when.** 26B-A4B is conditional with an explicit path or a typed
refuse. Dense Gemma 4, unified Gemma 4, and Qwen2 unchanged. No CUDA MoE.
No "Aptus supports MoE."

- [x] Lane 9 spec written and approved
- [x] Owner "go"
- [x] Topology + census + estimator
- [ ] Measured ladder or envelope refuse (conditional path shipped; operator ladder is a later new work dir)
- [ ] PR opened and merged

---

## Lane 10: Vision / multimodal SFT

Do not start until Lane 9 is closed and a Lane 10 spec is approved.

**Who owns what**

| Concern | Owner | Aptus will not |
| --- | --- | --- |
| Image/audio JSONL contract | Lane 10 spec | Grow plan statuses or bump 0.3 |
| Leakage / reverse-refuse (text rows on a vision path and the reverse) | Dataset facts | Silently train the unused tower |
| Language-tower text SFT | Lanes 6–7 | Break sanitize-drop of unused towers |
| Veriformis JSONL | Existing handoff | Import `.vfbundle` |

**Do**

1. Named increment with a hard non-goal: no new plan statuses, no 0.3
   bump, no referee growth.
2. Image/audio JSONL contract, leakage rules, and a refuse for text-only
   rows sent to a vision path (and the reverse).
3. One exact Gemma 4 pin plus one measured ladder, or an honest "vision
   path not executable yet."

**Done when.** Text SFT on the language tower still works. Vision is a
separate fail-closed path. Sanitize still drops unused towers on the text
path.

- [ ] Lane 10 spec written and approved
- [ ] Owner "go"
- [ ] Dataset-facts contract
- [ ] Measured ladder or honest not-executable
- [ ] PR opened and merged

---

## Parked track

Not the next lane. Owner must name a new program.

| Item | Why it is parked |
| --- | --- |
| Gemma 4 31B measured ladder | Already infeasible here (RAM and disk). Keep the no. |
| Qwen3 30B-A3B measured ladder | Compiler exists; envelope already failed. |
| CUDA semantic adapter reload | Required before a CUDA release claim. Not required to finish Gemma 4. |
| Aptus 0.2 product ship | One notarized identity exists for `edc6cfd`. Shipping is that HEAD plus a packet, not more features. |
| Multi-GPU / FSDP campaign | Needs its own protocol. M7-B never had a second host. |
| Journey B3 / six epochs / specialist | Operator use of Aptus. Not a product hole. |
| Chrome / Home / Models shell | Declined relative to Lane 4 craft. |

## Local hygiene (optional, not a lane)

- `.superpowers/mission-integrity-plan/STATUS.md` was talking like M9 / TP
  is in flight. Update it when touching remainder docs so a cold start
  does not resume the wrong program.
- `ROADMAP.md` last reviewed 2026-08-11. CUDA Phase 0–10 remains closed;
  do not rewrite that history when touching the date.

## Testing

- This filing: `PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest tests.aptus.test_documentation -v`
- Later lanes: the repository-wide checks in `CONTRIBUTING.md` before claiming done.
- Journey and Gemma 4 runtime evidence stays local and uncommitted.

## What "finished" means

The remainder concept is finished when:

1. 12B is no longer a false `no-policy-match`.
2. Lane 8 is skipped with a recorded census or implemented.
3. 26B-A4B is a visible conditional path or a typed refuse.
4. Vision is a fail-closed dataset path or an honest not-executable.
5. Dense Gemma 4, Qwen2, and the 0.2 referee are unchanged in contract.
6. No sentence on main says "Aptus supports Gemma 4 / MoE / vision."

That is family honesty. It is not a product ship and not quality.
