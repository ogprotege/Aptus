# Remainder Program Implementation Plan

> **Status:** Active | **Authority:** Implementation plan (subordinate to claim language and the remainder freeze) | **Applies to:** Aptus remainder program after Lane 6, not M10, not 0.3, not more 0.2 | **Audience:** Agents executing R0, then Lanes 7–10 one at a time | **Last reviewed:** 2026-08-26 | **Review by:** After Lane 7 starts, or when this remainder is superseded

> **For agentic workers:** Do not implement Lane 7–10 from this file alone. Each later lane needs its own spec freeze and an explicit owner "go." Use superpowers:subagent-driven-development or superpowers:executing-plans only inside one approved lane.

**Goal:** Keep Aptus fail-closed while closing the leftover Gemma 4 honesty gaps, one named increment at a time.

**Architecture:** R0 is documentation only. Lanes 7–9 are registry / inspect / plan-contract / compiler rows under family `gemma4`. Lane 10 is a new dataset-facts surface. The 0.2 referee (Facts → Compare → Compile → Validate → Run + last call) stays frozen.

**Tech Stack:** Existing Aptus Python core, MLX-LM bundles, claim-language docs. No OpenAPI hand-edits. No `aptus-work/` commits.

## Global Constraints

- Spec freeze: `docs/superpowers/specs/2026-08-26-remainder-program-design.md`
- Increment names are Lane 7–10. Not M10. Not 0.3. Not a 0.2 ship. Do not grow 0.2.
- One increment at a time. Spec → owner "go" → code + tests + docs → PR.
- Always open a GitHub PR after pushing Aptus work.
- Do not retrain Journey A, B, B2, E2B, or E4B. Do not overwrite those trees.
- Do not commit `aptus-work/` or `web/node_modules`.
- Do not force Gemma 4 31B or Qwen3 30B-A3B trains on this Mac.
- Do not start a Qwen3 MoE train inside a Gemma 4 increment.
- `measured-run-pass` is not quality. Gold is not quality. Recitation is not Use.
- Never hand-edit `docs/reference/openapi.v1.json` or `web/src/generated/openapi.ts`.

---

## Working order

```text
R0 freeze
  → Lane 7 (12B unified)  [Lane 8 only if 12B needs omit-v]
  → Lane 9 (26B-A4B MoE)
  → Lane 10 (vision)
  → stop, or owner names a new program (release / CUDA / huge-model host)
```

---

## R0: File the remainder

**Files:**
- Create: `docs/superpowers/specs/2026-08-26-remainder-program-design.md`
- Create: `docs/superpowers/plans/2026-08-26-remainder-program.md`
- Modify: `docs/maintenance/documentation-inventory.md`
- Modify: `docs/product/index.md`
- Modify: `docs/product/mission-integrity-plan.md`
- Modify: `CHANGELOG.md`
- Modify: `tests/aptus/test_documentation.py`

- [x] **Step 1: Write the freeze and this plan**
- [x] **Step 2: Inventory, navigation, changelog, and documentation-test counts**
- [x] **Step 3: Run** `PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest tests.aptus.test_documentation -v`
- [x] **Step 4: Stop.** Do not start Lane 7 code.

R0 is done when the two new Markdown files are approved, inventoried, and reachable. Lane 7 is not started.

---

## Lane 7: Gemma 4 unified (12B)

Do not start until a Lane 7 spec is approved and the owner says go.
Proposed spec: `docs/superpowers/specs/2026-08-31-lane-7-gemma4-unified-design.md`.
Completion plan: `docs/superpowers/plans/2026-08-31-remainder-completion.md`.

**Do**

1. Inspect the exact pin. Starting candidate: `mlx-community/gemma-4-12b-it-4bit` @ `73bcf090`, or the current immutable revision. Record architecture, layers, bits, KV pattern, and MoE-null facts.
2. Add a second exact identity under family `gemma4` for `gemma4_unified_text` / `Gemma4UnifiedForConditionalGeneration`. Do not alias it onto `model.gemma4.mlx.v1`.
3. Keep Qwen2 and dense Gemma 4 (`Gemma4ForConditionalGeneration`) paths unchanged.
4. Compare / compile / emit-run on this Mac. New work dir. Do not overwrite E2B/E4B.
5. If the envelope admits it: one measured ladder, pilot-required. If it refuses: keep the no with the RAM/disk ledger.

**Gate before train.** If 12B omits `v_proj` on k-equals-v layers, finish Lane 8 as a prerequisite task of Lane 7. If 12B is k-count equals v-count, skip Lane 8 for this pin.

**Done when.** Inspect is `ok` with compatibility **conditional**, not unsupported. 26B-A4B with real experts is still a visible no. One compile path exists. A `measured-run-pass` **or** a documented envelope refuse. Not quality.

- [ ] Lane 7 spec written and approved
- [ ] Owner "go"
- [ ] Inspect recorded
- [ ] Second exact identity (no alias)
- [ ] Compile / emit-run
- [ ] Measured ladder or envelope refuse
- [ ] PR opened and merged

---

## Lane 8: k-equals-v omitted-`v_proj`

Skip as a standalone PR if Lane 7 never needs it.

**Do**

1. Persist inspect-time per-target counts.
2. Allow omitted `v_proj` only for family `gemma4`, only when the layer contract is k-equals-v, and only when `k_proj` still appears at least once.
3. Parent verification still matches the bound census. No any-subset backslide for llama, qwen, or mistral.

**Done when.** Tests refuse asymmetric k/v, accept documented k-equals-v omit-v, and leave other families strict. No 31B train is required to close this increment.

- [ ] Needed by the Lane 7 pin, or deferred until a later pin needs it
- [ ] Spec (or Lane 7 spec section) approved
- [ ] Census tests
- [ ] PR opened and merged

---

## Lane 9: Gemma 4 26B-A4B MoE

Do not start until Lane 7 is closed and a Lane 9 spec is approved.

**Do**

1. Exact architecture plus immutable revision. Real expert integers stay MoE facts. Null experts stay dense.
2. Separate total versus active parameters. Resident weight is not active parameters.
3. Expert / router / adapter-target census. Reject accidental dense or zero-trainable.
4. Sparse memory/disk estimator that does not substitute active for resident.
5. One MLX QLoRA path, pilot-required. Compare stays an honest no until the row exists.
6. Measured ladder only if this Mac admits it. If not, conditional plus envelope refuse is success.

**Do not** start a Qwen3 30B-A3B train in this increment.

**Done when.** 26B-A4B is conditional with an explicit path or a typed refuse. Dense Gemma 4 and Qwen2 unchanged. No CUDA MoE. No "Aptus supports MoE."

- [ ] Lane 9 spec written and approved
- [ ] Owner "go"
- [ ] Topology + census + estimator
- [ ] Measured ladder or envelope refuse
- [ ] PR opened and merged

---

## Lane 10: Vision / multimodal SFT

Do not start until Lane 9 is closed and a Lane 10 spec is approved.

**Do**

1. Named increment with a hard non-goal: no new plan statuses, no 0.3 bump, no referee growth.
2. Image/audio JSONL contract, leakage rules, and a refuse for text-only rows sent to a vision path (and the reverse).
3. One exact Gemma 4 pin plus one measured ladder, or an honest "vision path not executable yet."

**Done when.** Text SFT on the language tower still works. Vision is a separate fail-closed path. Sanitize still drops unused towers on the text path.

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

---

## Local hygiene (optional, not a lane)

- `.superpowers/mission-integrity-plan/STATUS.md` still talks like M9 / TP is in flight. Stale.
- `ROADMAP.md` last reviewed 2026-08-11. CUDA Phase 0–10 remains closed; do not rewrite that history when touching the date.

---

## Testing

- R0: `PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest tests.aptus.test_documentation -v`
- Later lanes: the repository-wide checks in `CONTRIBUTING.md` before claiming done.
- Journey and Gemma 4 runtime evidence stays local and uncommitted.
