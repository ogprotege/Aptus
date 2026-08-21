# Lane 6 — Gemma 4 family admission

> **Status:** APPROVED 2026-08-21 — owner: "go dmanit" after locking approach 2 and section 1  
> **Authority:** Subordinate to claim language, current capabilities, and 0.2 cut-freeze  
> **Increment:** Lane 6. **Not M10.** **Not 0.3.** **Not a 0.2 patch.**  
> **Implementation plan:** this spec is the freeze; code follows in the same increment  
> **Last reviewed:** 2026-08-21  
> **Next scheduled review:** After the first measured Gemma 4 MLX ladder, or when vision / 26B-A4B slices open

Owner direction (chat 2026-08-21): Aptus must keep admitting new model
types. Gemma 4 inspect-unsupported is the trigger. Do not freeze the
family at one pin, one size, or four-bit. This Mac loads 4 / 6 / 8 / 1-bit.

---

## 1. Goal

Admit **Gemma 4** as an Aptus family. Inspect, Compare, Compile, and Run
use the pinned Hub revision's declared shape and bits. E2B, E4B, 12B, and
31B are in the family. Quantization bits 1–16 with a groupwise MLX layout
are inspect facts, not a four-bit religion.

`lmstudio-community/gemma-4-31B-it-MLX-8bit` is the same *class* as
`mlx-community/gemma-4-31b-it-8bit`. Inspect still uses a provider repo and
an immutable revision, not a local LM Studio folder.

## 2. Non-goals

- Growing the 0.2 referee (new plan statuses, new rank formula, 0.3 bump).
- Aliasing `gemma4` / `gemma4_text` onto the dense `gemma` (Gemma 2 / text-only Gemma 3) catalog.
- Copying Qwen2's "exactly 24 layers, exactly 4-bit" freeze onto Gemma 4.
- Auto-admitting every Hub architecture.
- Image / multimodal training data path (named next slice).
- 26B-A4B MoE (named next slice; Compare stays an honest no).
- Calling a passing train quality. Not overwriting Journey A/B. Not committing `aptus-work/`.

## 3. Slice 1 training surface

Text SFT on the language tower. Hub cards are `image-text-to-text`.
mlx-lm's `gemma4` wrapper trains `language_model` and drops vision/audio
weights at sanitize. Slice 1 does not grow an image JSONL path.

## 4. Catalog and inspect

| Provider | Aptus family |
|---|---|
| `gemma4` | `gemma4` |
| `gemma4_text` | `gemma4` |

Architecture in the exact policy identity:
`Gemma4ForConditionalGeneration`.

Inspect:

- Ignore MLX quantization metadata scalars such as `mode: affine`. Keep
  `bits` and `group_size`.
- Treat `enable_moe_block: false` and null expert counts as **dense**
  (`moe` is absent). Do not emit a contradictory MoE topology.
- Real expert integers remain MoE facts. The dense Gemma 4 policy then
  blocks with dense-topology-required until the MoE slice.

## 5. Compatibility policy

One policy, `model.gemma4.mlx.v1`, family `gemma4`.

Constraints:

- exact identity: family `gemma4`, model_type `gemma4_text`, architecture
  `Gemma4ForConditionalGeneration`
- dense: `moe` is null
- **no** layer-count freeze
- **no** four-bit freeze

Paths (unique path IDs; Qwen2 keeps `mlx-lm.qlora.single.dense-causal-lm.v1`):

- `mlx-lm.qlora.single.gemma4-dense.v1` — quantized-base LoRA
- `mlx-lm.lora.single.gemma4-dense.v1` — unquantized LoRA

Planner, not the identity constraint, picks the method:

- QLoRA when the pin declares bits 1–16 and a matching groupwise layout.
- LoRA when the pin is unquantized (bf16).
- QLoRA on an unquantized pin, or LoRA on a quantized pin, is unsupported
  with an explicit reason.

RAM/disk can still mark 31B bf16 infeasible. That is the envelope.

## 6. Compiler and bundle

- Catalog and `plan_contract.MODEL_TARGET_MODULES` include `gemma4` dense
  q/k/v/o/gate/up/down.
- MLX QLoRA candidate quantization is `mlx-{bits}bit-groupwise` from the
  pin. Qwen2 4-bit remains `mlx-4bit-groupwise`.
- Bundle `train.py` QLoRA checks declared bits 1–16 against the plan, not
  `bits == 4`.
- Pinned `config.json` may contain null MoE keys; validation treats null
  as "not declared."
- Layout compare ignores `mode` / other scalar metadata.

## 7. Evidence

New implementation-reviewed record `policy.gemma4.mlx.v1`. It does **not**
transfer Path Alpha measured-run-pass. Every Gemma 4 pin stays
pilot-required until its own ladder.

First measured ladder: a small 4-bit instruct pin (E2B or E4B) on this Mac.
31B is in the policy from day one; it is not hidden until that ladder.

## 8. Done when

Inspect of `mlx-community/gemma-4-31b-it-4bit` and the 8-bit pin is
`ok` with compatibility **conditional**, not unsupported.
E2B 4-bit is the same family. 26B-A4B with real experts is still a visible
no. Tests cover inspect mapping, dense-null MoE, affine `mode`, 4-bit and
8-bit path match, and Qwen2 4-bit unchanged.
