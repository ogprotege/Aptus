# M9 — Sustain audit of M0–M8

> **Status:** COMPLETE 2026-08-13  
> **HEAD audited:** `fc5186b843ba5ab8f432df2bb3697d58f308018e` (Merge #93)  
> **Kind:** Check of work already done. Not a new method, host, or measured ladder.

## 16.2 Standing checklist

| Question | Verdict | Evidence |
| --- | --- | --- |
| Does this make false-yes more or less likely? | **Less likely** than M0 | Claim policy, refusal catalog, eval/loss split, Compare correction non-claim |
| Does claim language still match evidence? | **Pass after wording fix** | Packets exist; live “current HEAD” for Path Beta was stale at `fc5186b` and is rewritten to recorded source |
| Did we hide a rejection? | **No** | All 12 planner rows remain; 422 keeps rejected candidates; M7-B skip is explicit |
| Did we add a method without compiler + gates? | **No** | Selectable set still Full / LoRA / int8-LoRA / QLoRA with compiler IDs |
| Did we expand a claim without a packet? | **No transfer claims** | M6/M7-A/M7-C/M8 stay packet-bound; README omitted those rows (fixed in this phase) |

## 16.3 Stop list

| Stop | Held? |
| --- | --- |
| Universal method recommendation across undocumented models | Yes |
| Silent dependency installation | Yes (`installation_performed: false`) |
| MLX resume without full state contract | Yes (`--resume-from` rejected) |
| Remote multi-user job service without auth | Yes (loopback + session token; no tenants) |
| “AI agent trains for you” bypassing the ladder | Yes (no such surface) |

## 16.4 Retention

`docs/operations/state-storage-retention.md` still has no auto-cleanup command. Dated 2026-08-1* packets publish digests and summaries, not raw job logs or secrets. Historical 2026-07-27 MLX packet still contains in-tree logs; it is frozen evidence, not a new leak.

## Residuals parked (not false-yes)

- Compare heading “safe plan” and unlabeled Pareto cell can be misread; body already disclaims optimality.
- Objective enum name `quality` is ranking, not model quality.
- Path packets print job IDs / hostname; campaign packets redact those. Not secrets.
- M7-A packet is thinner than M3/M4. Identities still bind.
- Measured Path Alpha/Beta ladders are **not** re-proved at `fc5186b`. M9 does not invent a current-HEAD transfer. Use recorded source commits.

## P-15

M1 assigned README claim-language sustain to M9. Re-audit found no live false-yes on README. Drift was omission (M8 CLI, mission-path packets) and “current HEAD” rot. Closed by this audit plus the standing PR checklist.
