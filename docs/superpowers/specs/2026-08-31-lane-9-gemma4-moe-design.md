# Lane 9 — Gemma 4 26B-A4B MoE

> **Status:** APPROVED 2026-08-31 — owner: "go"
> **Authority:** Subordinate to claim language, current capabilities, the 0.2 cut-freeze, and the remainder freeze
> **Increment:** Lane 9. **Not M10.** **Not 0.3.** **Not a 0.2 patch.** **Not an alias of Lane 6.**
> **Parent freeze:** `docs/superpowers/specs/2026-08-26-remainder-program-design.md`
> **Implementation plan:** `docs/superpowers/plans/2026-08-31-remainder-completion.md`
> **Last reviewed:** 2026-08-31
> **Next scheduled review:** After this increment merges, or when this proposal is superseded

Owner sign-off: "go" (chat 2026-08-31), after Lane 7 merged as PR #111.
Bound MLX-LM 0.31.3 loads this architecture through `gemma4_text` with
`enable_moe_block`, so this increment is Exit A: a conditional
pilot-required path, not a compiler-contract refuse. It does not admit
Qwen3 MoE evidence, CUDA MoE, vision, or quality.

---

## 1. Goal

Close the **false no** on Gemma 4 26B-A4B.

Lane 6 named 26B-A4B as later. Live inspect of the Hub pin today maps
`gemma4_text` / `Gemma4ForConditionalGeneration` onto family `gemma4`.
The dense policy then blocks on `dense-topology-required` because the pin
declares real experts. That is Aptus pretending a sparse Gemma 4 identity
is a broken dense one.

Lane 9 adds a **new topology identity** `gemma4_moe`. It is not a size
tweak of `model.gemma4.mlx.v1`. It does not grow the 0.2 referee.

## 2. Why this is a separate family

| Pin class | Provider type | Architecture | Topology | Live policy |
| --- | --- | --- | --- | --- |
| Dense E2B / E4B / 31B language tower | `gemma4_text` | `Gemma4ForConditionalGeneration` | `moe` null | `model.gemma4.mlx.v1` (Lane 6) |
| Unified 12B | `gemma4_unified_text` | `Gemma4UnifiedForConditionalGeneration` | dense | `model.gemma4-unified.mlx.v1` (Lane 7 Exit B) |
| 26B-A4B MoE | `gemma4_text` | `Gemma4ForConditionalGeneration` | 128 experts, top-8 | none (this lane) |

The 26B pin **reuses the dense provider type and architecture**. The
differentiator is declared MoE topology (`enable_moe_block=true`,
`num_experts=128`, `top_k_experts=8`). A second `gemma4` policy row that
claimed architecture or `gemma4_text` would steal dense subjects
(first-eligible-wins) or steal 26B into `dense-topology-required`.

Inspect must therefore route real Gemma 4 MoE to family `gemma4_moe`
before policy evaluation. The MoE policy claims **family only**. Dense
Gemma 4 keeps claiming architecture / `gemma4_text` / `gemma4`. Sparse
identity markers (`moe` in family) skip dense rows.

Do **not** alias 26B onto `model.gemma4.mlx.v1`. Do not map it onto
`qwen3_moe`. Prefix matching is still forbidden.

## 3. Exact pin (starting candidate)

Live Hub config at Lane 9 start (2026-08-31):

- Repo: `mlx-community/gemma-4-26b-a4b-it-4bit`
- Revision: `0d77464eeb233a2da68ebf9d7dc4edaac7db956d`
- Architecture: `Gemma4ForConditionalGeneration`
- Outer `model_type`: `gemma4`
- Text `model_type`: `gemma4_text`
- 30 layers, hidden 2816, intermediate 2112, moe intermediate 704
- `enable_moe_block`: true
- `num_experts`: 128
- `top_k_experts`: 8
- `attention_k_eq_v`: true
- `num_kv_shared_layers`: 0
- `layer_types`: 25 `sliding_attention` + 5 `full_attention`
- 4-bit group 64 default, 8-bit
  `language_model.model.layers.{0..29}.router.proj`
- Vision/audio configs present; Lane 9 is language-tower text SFT.
  Sanitize still drops unused towers. Lane 10 is vision.

Do not silently retarget. Do not overwrite
`aptus-work/gemma4-e2b-v4-run` or `aptus-work/gemma4-e4b-v3-run`. New
work dir if a measured ladder is later admitted.

## 4. Runtime honesty (mlx-lm 0.31.3)

`mlx_lm.models.gemma4_text` already implements:

- `enable_moe_block` DecoderLayer with dense MLP **and** `Router` +
  `Experts` (`SwitchGLU`)
- `Router.proj`: `Linear(hidden_size, num_experts)`
- `use_k_eq_v = attention_k_eq_v and not is_sliding` (omits `v_proj` on
  full-attention layers)
- `quant_predicate`: 8-bit `router.proj`
- sanitize maps `experts.gate_up_proj` / `down_proj` to SwitchGLU

There is no missing loader. Exit B (compiler-contract) is the wrong
exit for this pin.

| Exit | When | Inspect compatibility | What Aptus may say |
| --- | --- | --- | --- |
| **A — executable** | Bound mlx-lm loads this architecture without alias | `conditional` | "conditional on a target-host pilot" |
| **B — typed refuse** | Bound mlx-lm cannot load this architecture | `unsupported` with a **compiler-contract** reason, never `no-policy-match` | "unsupported by the current compiler contract" |

This pin is Exit A. Compare / compile may emit a bundle. emit-run on
this Mac is measured ladder **or** envelope refuse. Both close the
lane. Neither is quality. Do not call Exit A "Aptus supports MoE."

## 5. Proposed identity

| Field | Value |
| --- | --- |
| Aptus family | `gemma4_moe` |
| Provider `model_type` | `gemma4_text` (kept raw) |
| Architecture | `Gemma4ForConditionalGeneration` |
| Policy ID | `model.gemma4-moe.mlx.v1` |
| Policy version | `1.0.0` |
| QLoRA path | `mlx-lm.qlora.single.gemma4-moe.v1` |
| LoRA path | none (this pin is quantized; unquantized LoRA is not this lane) |
| Evidence ID | `policy.gemma4-moe.mlx.v1` |
| Adapter profile | `attention-qkvo.v1` (`q_proj`, `k_proj`, `v_proj`, `o_proj`) |
| Topology | sparse: 128 experts, 8 per token, no shared expert, `decoder_sparse_step` default 1, `mlp_only_layers` default `[]` |
| Bits | freeze 4-bit default + 8-bit `model.layers.{layer}.router.proj` |
| Layers | inspect-declared; no 30-layer freeze |

Inspect mapping, architecture-guarded:

```text
gemma4_text + Gemma4ForConditionalGeneration + declared MoE
  → family gemma4_moe
gemma4_text + Gemma4ForConditionalGeneration + moe null
  → family gemma4 (Lane 6, unchanged)
gemma4_unified_text + Gemma4UnifiedForConditionalGeneration
  → family gemma4 (Lane 7, unchanged)
```

Null experts stay dense. Real expert integers stay MoE facts.

Policy claims **family `gemma4_moe` only**. Exact-identity still names
all three fields. Do not claim architecture or `gemma4_text`; those
values are shared with dense Gemma 4.

Path ID must be unique. Do not reuse
`mlx-lm.qlora.single.attention-qkvo.v1` (Qwen3) or
`mlx-lm.qlora.single.gemma4-dense.v1`.

## 6. Inspect facts this pin must persist

In addition to the Lane 6 dense fields:

- `moe.expert_count` from `num_experts`
- `moe.experts_per_token` from `num_experts_per_tok` **or**
  `top_k_experts` (Gemma 4 name)
- `moe.expert_intermediate_size` from `moe_intermediate_size`
- `moe.decoder_sparse_step` default `1` when Gemma-style MoE omits it
- `moe.mlp_only_layers` default `[]` when Gemma-style MoE omits it
- `moe.shared_expert_intermediate_size` null (no shared expert)
- `attention_k_eq_v`
- `num_kv_shared_layers`
- quantization layout with `language_model.` prefix stripped so Hub
  `language_model.model.layers.N.router.proj` becomes
  `model.layers.N.router.proj`

Do not invent `num_experts_per_tok` on the wire. Persist Aptus
`experts_per_token`. Qwen3 `num_experts_per_tok` still wins when
present.

## 7. Lane 8 (prerequisite of 26B train, same increment)

The 12B unified pin skipped Lane 8 (no loader, `attention_k_eq_v=false`).
This 26B pin needs it.

mlx-lm census for this pin:

- `k_proj` on all 30 layers (`num_kv_shared_layers=0`)
- `v_proj` only on the 25 sliding layers
- expected `k_count=30`, `v_count=25`

`mlx_trainable_target_instance_total` today requires `k_count == v_count`.
Lane 8 allows omitted `v_proj` only when:

- family is `gemma4` or `gemma4_moe`
- `k_proj` still appears at least once
- `v_count <= k_count` (omit v, never omit k while keeping v)
- both counts stay in `1..layers` for planned targets
- q/o (and dense MLP targets on family `gemma4`) still cover every layer

Do not weaken llama, qwen, mistral, or `qwen3_moe`. Asymmetric k/v that
is not this documented omit remains a refuse.

## 8. Compiler and bundle (Exit A)

- New path ID `mlx-lm.qlora.single.gemma4-moe.v1`.
- Adapter targets: attention q/k/v/o only. Do not train 128 experts.
  Router and expert weights stay frozen.
- Catalog family `gemma4_moe` uses `attention-qkvo.v1` targets.
- `MLX_SPARSE_ADAPTER_FAMILIES` includes `gemma4` and `gemma4_moe`.
- QLoRA uses the frozen 4-bit + 8-bit `router.proj` layout.
- `plan_contract` override pricing accepts unique
  `model.layers.*.mlp.gate` **or** `model.layers.*.router.proj`. Qwen3
  `mlp.gate` stays valid.
- Architecture-contract MoE identity accepts reviewed Qwen3 **or**
  reviewed Gemma 4 MoE. A Gemma 4 MoE plan must not be rejected as
  "requires the exact reviewed Qwen3 MoE identity."
- Packed-checkpoint leftover-tower exclusion stays. Do not treat Hub
  vision bytes as container overhead.
- Sanitize still drops unused vision/audio / projector payloads on the
  **text** path. Lane 9 is not Lane 10.
- Dense Gemma 4, unified Gemma 4, Qwen2, and Qwen3 MoE rows stay
  bitwise-behavior unchanged except the Lane 8 census relaxation for
  Gemma 4 families.

## 9. Estimator honesty

Total resident parameters stay total. Derived `active_parameters` describe
per-token routing:

```text
inactive = sparse_layers * (expert_count - experts_per_token)
         * 3 * hidden_size * expert_intermediate_size
active   = parameters - inactive
```

The 26B hybrid layer still has a dense MLP (h1) plus routed experts (h2).
Dense MLP weights are always resident and always active; they stay inside
`parameters` and are not subtracted. Storage, metadata, staging, and disk
use **total** parameters. Active parameters never substitute for resident
weights.

Router override pricing: `hidden_size * expert_count` per
`router.proj` (same shape as Qwen3 `mlp.gate`).

## 10. Evidence

New implementation-reviewed record `policy.gemma4-moe.mlx.v1`. It does
**not** transfer E2B/E4B `measured-run-pass`. It does not transfer Path
Alpha. It does not transfer Qwen3 30B-A3B envelope refuse into a Gemma 4
yes or no.

A later `measured-run-pass` on this pin is artifact-scoped. Gold and
training loss are not quality. Recitation is not Use.

This increment ships the conditional path and honest envelope. A measured
ladder on this Mac runs only if the ledger admits it, in a **new** work
dir, and is not claimed by merging the policy row. Envelope refuse is
success for the lane.

## 11. Non-goals

- Growing the 0.2 referee.
- Naming this M10 or 0.3.
- Auto-admitting every Hub architecture.
- Aliasing 26B onto dense `model.gemma4.mlx.v1`, unified, or `gemma`.
- CUDA MoE. "Aptus supports MoE."
- Qwen3 30B-A3B train. Gemma 4 31B train on this Mac.
- Training 128 experts or adapter-targeting `router.proj` / SwitchGLU.
- Lane 10 vision JSONL.
- Overwriting Journey A/B/B2/E2B/E4B. Committing `aptus-work/`.
- Substituting active parameters for resident weight in the envelope.

## 12. Claim language

Use: "named remainder increment"; "exact Gemma 4 MoE identity";
"conditional on a target-host pilot"; "envelope refuse";
"recommended within the enumerated candidate set";
"resident weight is not active parameters."

Do not use: "Aptus supports Gemma 4"; "Aptus supports MoE";
"26B is the same as E2B"; "reviewed 26B identity" before this pin's
own ladder or envelope refuse; "quality yes"; "0.3"; "M10."

## 13. Done when

1. Owner has approved this spec and said go.
2. The 26B pin's inspect is `ok`, family `gemma4_moe`, and is **not**
   `no-policy-match` and **not** `dense-topology-required`.
3. Dense `Gemma4ForConditionalGeneration` with `moe` null still matches
   `model.gemma4.mlx.v1` only.
4. Unified 12B still matches `model.gemma4-unified.mlx.v1` only
   (compiler-contract).
5. Qwen3 MoE still matches `model.qwen3-moe.mlx-qlora` only.
6. Lane 8 omit-v census is recorded for family `gemma4` / `gemma4_moe`.
7. The estimator prices total parameters for residency. Active
   parameters stay derived and smaller.
8. Exit A path is conditional and pilot-required, **or** the host
   envelope refuses. Gold cannot close this lane.
9. Tests and docs ship in the same PR. PR is opened, CI is fully green,
   and the merge is a merge commit. `aptus-work/` is not in the commit.
10. Lane 10 is not started.
