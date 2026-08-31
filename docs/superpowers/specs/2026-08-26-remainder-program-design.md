# Remainder program — freeze

> **Status:** APPROVED 2026-08-26 — owner: "File it."
> **Authority:** Subordinate to claim language, current capabilities, and the 0.2 cut-freeze
> **Increment:** Remainder program (R0). Names Lanes 7–10. **Not M10.** **Not 0.3.** **Not a 0.2 patch.**
> **Implementation plan:** `docs/superpowers/plans/2026-08-26-remainder-program.md`
> **Last reviewed:** 2026-08-26
> **Next scheduled review:** After the Lane 7 spec is approved, or when a parked-track program opens

Owner sign-off (chat 2026-08-26): file the remainder as named increments after
Lane 6. Do not grow the 0.2 referee. Do not start Lane 7 code from this freeze
alone.

---

## 1. Goal

Sequence the leftover work so Aptus stays honest about the family it already
opened, and so a later increment cannot silently become more 0.2.

Mission first: trust the no, prove the yes one exact family/runtime row at a
time. Owner direction still holds: keep admitting new types, do not shrink
Gemma 4 to E2B/E4B, one exact identity per increment.

This freeze names the program. It does not admit 12B, MoE, or vision. Each
later lane still needs its own spec and an explicit owner "go" before code.

## 2. Why this order

Lane 6 §1 says E2B, E4B, 12B, and 31B are in the family. The live policy
identity is only `gemma4_text` / `Gemma4ForConditionalGeneration`. The 12B Hub
pin is `gemma4_unified_text` / `Gemma4UnifiedForConditionalGeneration`. That is
a promised size that today comes back unsupported. Close that false no first.

Named Lane 6 leftovers after that: k-equals-v omitted-`v_proj`, then 26B-A4B
MoE, then a vision JSONL path. Huge-model trains and CUDA release work stay
parked. This Mac already refused Gemma 4 31B and Qwen3 30B-A3B on envelope.

## 3. Closed and not reopened

| Item | Disposition |
| --- | --- |
| CUDA campaign Phases 0–10 | Closed. No Phase 11. |
| Mission Integrity M0–M9 | Closed. No M10. M7-B skipped. |
| TP, Lane 3, 0.2 cut-freeze, Lane 4, Lane 6 | Closed on main. |
| Lane 5 first train (Journey B2) | Closed. Train recitation 13/32. Specialist not claimed. |
| Journeys A, B, B2, E2B, E4B | Do not retrain. Do not overwrite. |

The Lane 5 six-epoch recipe applied only if train recitation was 0/32. It was
not. Further recitation trains are operator journeys, not Aptus phases.

`qwen3_moe` already has a compiler row. The recorded 30B-A3B attempt stopped
on envelope. That is a trustworthy no, not missing admission code. Do not
start a Qwen3 MoE train inside a Gemma 4 increment.

## 4. Named increments

```text
R0 freeze (this document)
  → Lane 7  Gemma 4 unified (12B)
  → Lane 8  k-equals-v omitted-v_proj (only if a pin needs it)
  → Lane 9  Gemma 4 26B-A4B MoE
  → Lane 10 vision / multimodal SFT
  → stop, or owner names a new program
```

One increment at a time. Spec approved → implement → PR. Always open the PR
after push. Do not commit `aptus-work/`.

### Lane 7 — Gemma 4 unified (12B)

Second exact identity under family `gemma4`. Do not alias
`gemma4_unified_text` / `Gemma4UnifiedForConditionalGeneration` onto the dense
E2B/E4B policy. Inspect the exact Hub pin (starting candidate
`mlx-community/gemma-4-12b-it-4bit` @ `73bcf090`, or the current immutable
revision). Compare / compile / emit-run. Measured ladder if the envelope
admits it; otherwise keep the no with the ledger.

If 12B omits `v_proj` on k-equals-v layers, Lane 8 is a prerequisite task of
Lane 7. If 12B is k-count equals v-count (KV-shared omit-both, like E2B), skip
Lane 8 for that pin.

### Lane 8 — k-equals-v omitted-`v_proj`

`ece3e51` already requires `k_proj` count equal `v_proj` count. This slice
allows omitted `v_proj` only for family `gemma4`, only when the layer contract
is k-equals-v, and only when `k_proj` still appears at least once. Do not
weaken the census for llama, qwen, or mistral. Skip as a standalone PR if
Lane 7 never needs it.

### Lane 9 — Gemma 4 26B-A4B MoE

New topology, not a size tweak. Exact architecture and immutable revision.
Separate total versus active parameters. Resident weight is not active
parameters. One MLX QLoRA path, pilot-required. Measured ladder only if this
Mac admits it. A passing Gemma 4 MoE path does not admit Qwen3 MoE evidence.

### Lane 10 — Vision / multimodal SFT

New dataset facts. Hard non-goal: no new plan statuses, no 0.3 bump, no
referee growth. Text SFT on the language tower must keep working.

## 5. Parked track

Pull any of these only as a separate named program, after the owner asks.

| Item | Why it is parked |
| --- | --- |
| Gemma 4 31B measured ladder | Already infeasible here (RAM and disk). Keep the no. |
| Qwen3 30B-A3B measured ladder | Compiler exists; envelope already failed. |
| CUDA semantic adapter reload | Required before a CUDA release claim. Not required to finish Gemma 4. |
| Aptus 0.2 product ship | One notarized identity exists for `edc6cfd`. Shipping is that HEAD plus a packet, not more features. |
| Multi-GPU / FSDP campaign | Needs its own protocol. M7-B never had a second host. |
| Journey B3 / six epochs / specialist | Operator use of Aptus. Not a product hole. |
| Chrome / Home / Models shell | Declined relative to Lane 4 craft. |

## 6. Non-goals

- Growing the 0.2 referee (new plan statuses, new rank formula, 0.3 bump).
- Naming any of this M10.
- Auto-admitting every Hub architecture.
- Aliasing `gemma4` onto dense `gemma`, or unified onto the Lane 6 identity.
- Calling `measured-run-pass`, gold, or recitation a quality yes.
- Overwriting Journey A/B/B2 or the E2B/E4B trees.
- Committing `aptus-work/`.
- Two families, two runtimes, or two topologies in one PR.
- Forcing a 31B or 30B-A3B train on this Mac.

## 7. Claim language

Use: "named remainder increment"; "conditional on a target-host pilot";
"unsupported by the current compiler contract"; "envelope refuse";
"recommended within the enumerated candidate set."

Do not use: "Aptus supports Gemma 4"; "Aptus supports MoE"; "12B is the same
as E2B"; "reviewed 12B identity" before that pin's own ladder; "specialist";
"quality yes"; "0.3."

## 8. Done when

This remainder freeze is done when this file and the matching plan are
approved, inventoried, and reachable. Lane 7 is not started.

Lane 7–10 are each done only under their later spec. A measured-run-pass or a
documented envelope refuse can close a lane. Gold and training loss cannot.
