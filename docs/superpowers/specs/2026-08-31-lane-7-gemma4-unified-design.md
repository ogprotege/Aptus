# Lane 7 — Gemma 4 unified (12B)

> **Status:** PROPOSED 2026-08-31 — no owner "go." Do not start Lane 7 code from this file.
> **Authority:** Subordinate to claim language, current capabilities, the 0.2 cut-freeze, and the remainder freeze
> **Increment:** Lane 7. **Not M10.** **Not 0.3.** **Not a 0.2 patch.** **Not an alias of Lane 6.**
> **Parent freeze:** `docs/superpowers/specs/2026-08-26-remainder-program-design.md`
> **Implementation plan:** `docs/superpowers/plans/2026-08-31-remainder-completion.md`
> **Last reviewed:** 2026-08-31
> **Next scheduled review:** After owner approval, or when this proposal is superseded

Owner sign-off required before code. This file names the second exact
Gemma 4 identity. It does not admit 12B, MoE, or vision by existing.

---

## 1. Goal

Close the **false no** on Gemma 4 12B.

Lane 6 §1 already said E2B, E4B, 12B, and 31B are in the family. The live
policy identity is only `gemma4_text` / `Gemma4ForConditionalGeneration`.
The 12B Hub pin is a **different architecture**. Today inspect returns
`unsupported` / `no-policy-match`. That is Aptus pretending not to know a
type it already named.

Lane 7 adds a **second exact identity** under family `gemma4`. It does not
grow the 0.2 referee. It does not shrink Gemma 4 to E2B/E4B.

## 2. Why this is a separate identity

| Pin class | Provider type | Architecture | Live policy |
| --- | --- | --- | --- |
| Dense E2B / E4B / 31B language tower | `gemma4_text` | `Gemma4ForConditionalGeneration` | `model.gemma4.mlx.v1` (Lane 6) |
| Unified 12B | `gemma4_unified_text` | `Gemma4UnifiedForConditionalGeneration` | none (this lane) |

Do **not** alias unified onto `model.gemma4.mlx.v1`. Do not map
`gemma4_unified_text` onto dense `gemma`. Prefix matching is still
forbidden.

## 3. Exact pin (starting candidate)

Recorded local inspect (uncommitted, 2026-08-21):

- Repo: `mlx-community/gemma-4-12b-it-4bit`
- Revision: `73bcf09092aa277861d5a191b989b666f7f32e8f`
- Inspect `status: ok`
- Compatibility today: `unsupported` / `no-policy-match`
- Facts: 48 layers, hidden 3840, intermediate 15360, 16 heads, 8 KV heads,
  context 262144, vocab 262144, 4-bit groupwise 64, `moe: null`

Re-inspect at Lane 7 start. If Hub moved, pin the **current immutable
revision** and write it down. Do not silently retarget. Do not overwrite
`aptus-work/gemma4-e2b-v4-run` or `aptus-work/gemma4-e4b-v3-run`. New work
dir.

## 4. Runtime honesty (mlx-lm)

The Aptus test venv's mlx-lm currently ships:

- `mlx_lm/models/gemma4.py` (`model_type = "gemma4"`, wraps `gemma4_text`)
- `mlx_lm/models/gemma4_text.py`

It does **not** ship `gemma4_unified`. Transformers does. Loading 12B by
pretending it is `Gemma4ForConditionalGeneration` is the alias this lane
exists to forbid.

Lane 7 therefore has two honest exits. Both close the false no. Neither is
quality.

| Exit | When | Inspect compatibility | What Aptus may say |
| --- | --- | --- | --- |
| **A — executable** | Bound mlx-lm loads this architecture without alias | `conditional` | "conditional on a target-host pilot" |
| **B — typed refuse** | Bound mlx-lm cannot load this architecture | `unsupported` with a **compiler-contract** reason, never `no-policy-match` | "unsupported by the current compiler contract" |

Exit A still needs compare / compile / emit-run on this Mac, then a
measured ladder **or** an envelope refuse (RAM/disk). Exit B is success
for this increment if the reason is typed and the dense Lane 6 path is
unchanged. Do not call Exit B "Aptus supports 12B."

## 5. Proposed identity

| Field | Value |
| --- | --- |
| Aptus family | `gemma4` |
| Provider `model_type` | `gemma4_unified_text` (kept raw) |
| Architecture | `Gemma4UnifiedForConditionalGeneration` |
| Policy ID | `model.gemma4-unified.mlx.v1` |
| Policy version | `1.0.0` |
| QLoRA path | `mlx-lm.qlora.single.gemma4-unified.v1` |
| LoRA path | `mlx-lm.lora.single.gemma4-unified.v1` |
| Evidence ID | `policy.gemma4-unified.mlx.v1` |
| Adapter profile | `dense-causal-lm.v1` (q/k/v/o/gate/up/down) |
| Topology | dense: `moe` is null |
| Bits | inspect-declared 1–16 groupwise; no four-bit freeze |
| Layers | inspect-declared; no 24-layer freeze |

Inspect mapping, Gemma-3 style, architecture-guarded:

```text
gemma4_unified_text + Gemma4UnifiedForConditionalGeneration → family gemma4
gemma4_unified_text + any other architecture → do not map
```

Dense mapping stays:

```text
gemma4 / gemma4_text + Gemma4ForConditionalGeneration → model.gemma4.mlx.v1
```

## 6. Inspect facts this pin must persist

In addition to the Lane 6 dense fields, persist and show:

- `attention_k_eq_v`
- `num_kv_shared_layers`
- loaded `k_proj` instance count
- loaded `v_proj` instance count

Those four decide Lane 8. They are not optional comments.

mlx-lm `gemma4_text.Attention`:

- KV-shared layers (`layer_idx >= N - num_kv_shared_layers`) omit **both**
  `k_proj` and `v_proj` (`has_kv = False`). Lane 6 already allows that for
  family `gemma4` when the two counts match.
- `attention_k_eq_v` on full-attention layers omits **only** `v_proj` and
  reuses `k_proj` (`use_k_eq_v`). That is Lane 8. Transformers' unified
  text default is `attention_k_eq_v = False`, `num_kv_shared_layers = 0`.
  **Do not trust the default.** Read the pin.

## 7. Lane 8 gate (before any 12B train)

| Observed census | Action |
| --- | --- |
| `k_proj` count == `v_proj` count (including KV-shared omit-both) | Skip Lane 8 for this pin |
| `v_proj` omitted on k-equals-v layers, `k_proj` still present at least once | Lane 8 is a prerequisite task of Lane 7 |
| Asymmetric k/v that is not the documented k-equals-v omit | Refuse. Do not weaken llama/qwen/mistral |

No 12B train until this gate is recorded.

## 8. Compiler and bundle (Exit A only)

- New path IDs. Do not reuse `mlx-lm.qlora.single.gemma4-dense.v1`.
- Sanitize still drops unused vision/audio / projector payloads on the
  **text** path. Lane 7 is not Lane 10.
- QLoRA uses declared bits 1–16 (`mlx-{n}bit-groupwise`).
- Packed-checkpoint leftover-tower exclusion stays. Do not treat Hub
  vision bytes as container overhead.
- Qwen2 and dense Gemma 4 rows stay bitwise-behavior unchanged.

## 9. Evidence

New implementation-reviewed record `policy.gemma4-unified.mlx.v1`. It does
**not** transfer E2B/E4B `measured-run-pass`. It does not transfer Path
Alpha.

A later `measured-run-pass` on this pin is artifact-scoped. Gold and
training loss are not quality. Recitation is not Use.

## 10. Docs honesty leftover from Lane 6

`docs/product/current-capabilities.md` still does not name Gemma 4.
`docs/reference/capability-matrix.md` still says "Gemma remains
license-excluded" in the CUDA-campaign sense.

Lane 7 docs (and the remainder-completion plan's hygiene task) must
distinguish, in the same change as behavior:

- Dense Gemma 4 MLX LoRA/QLoRA is Lane 6, conditional, pilot-required, not
  CUDA, not quality.
- CUDA Gemma 2/3 license-exclusion is unchanged.
- Unified 12B is this increment, Exit A or Exit B as recorded.

## 11. Non-goals

- Growing the 0.2 referee.
- Naming this M10 or 0.3.
- Auto-admitting every Hub architecture.
- Aliasing unified onto dense `model.gemma4.mlx.v1` or onto `gemma`.
- 26B-A4B MoE (Lane 9). Vision JSONL (Lane 10).
- Qwen3 30B-A3B train. Gemma 4 31B train on this Mac.
- Overwriting Journey A/B/B2/E2B/E4B. Committing `aptus-work/`.
- "Aptus supports Gemma 4." "12B is the same as E2B." "reviewed 12B
  identity" before this pin's own ladder or typed refuse.
- Shipping a loader that Aptus does not own. If mlx-lm lacks unified,
  Aptus refuses; it does not vendor a fork in this increment.

## 12. Claim language

Use: "named remainder increment"; "second exact identity under family
gemma4"; "conditional on a target-host pilot"; "unsupported by the current
compiler contract"; "envelope refuse"; "recommended within the enumerated
candidate set."

Do not use: "Aptus supports Gemma 4"; "Aptus supports 12B"; "unified is
dense Gemma 4"; "quality yes"; "0.3"; "M10."

## 13. Done when

1. Owner has approved this spec and said go.
2. The 12B pin's inspect is `ok` and is **not** `no-policy-match`.
3. Dense `Gemma4ForConditionalGeneration` still matches
   `model.gemma4.mlx.v1` only.
4. 26B-A4B with real experts is still a visible no.
5. Exit A **or** Exit B is recorded. Gold cannot close this lane.
6. Lane 8 is either skipped with a recorded census or finished first.
7. Tests and docs ship in the same PR. PR is opened, CI is fully green,
   and the merge is a merge commit. `aptus-work/` is not in the commit.
