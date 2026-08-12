# M2 — COMPLETION NOTE

> **Phase status:** **COMPLETE**

| Field | Value |
| --- | --- |
| Phase | M2 |
| Title | Trust the “no” (refusal integrity) |
| Started (UTC) | 2026-08-12T00:01:14Z |
| Completed (UTC) | 2026-08-12 |
| Base commit | `3ebba7e` (main) + M0/M1 workspace merge from PR #85 line |
| Suite | 951 Python OK · 134 web OK · Ruff · Typecheck |

## Mission check

- Live P0 false-yes? **none** (M1 had 0; none introduced)
- Conditional painted as success? **no** — labels use `conditional · pilot required`
- Multi-GPU single-device looking ready? **no** — `unsupported · not runtime-ready` + guidance
- Experimental methods selectable? **no** (unchanged)

## Deliverables

| Item | Path |
| --- | --- |
| Refusal catalog | `.superpowers/mission-integrity-plan/M2-refusal-catalog.md` |
| Python guidance module | `src/aptus/refusal.py` |
| CLI stderr guidance | `src/aptus/cli.py` (`spec-plan` / `plan` / `build`) |
| Web mapping + labels | `web/src/lib/refusal.ts`, Compare/FitLedger/CandidateComparison |
| Tests | `tests/aptus/test_refusal.py`, `web/src/lib/refusal.test.ts` |
| Troubleshooting top refusals | `docs/guides/troubleshooting.md` |

## M1 P1 owned by M2

| ID | Disposition |
| --- | --- |
| P-18 | **Closed** — structured what/why/what-can-change |
| P-03 | **Closed** — catalog + UI/CLI completeness for plan refusals |
| P-12 | **Closed** for presentation — multi-GPU single-device labels + guidance |
| P-01 polish | Addressed via fit labels |
| P-02 CLI | Addressed via stderr refusal block |

## Explicit non-claims

- Does not re-prove Path Alpha/Beta at HEAD (M3/M4)
- Does not add OpenAPI field (presentation layer only; plan identity pure)
- Does not invent new methods or AutoML corrections

## Next phase allowed?

- **Yes:** M3 Path Alpha only (needs cost/authorization for measured work)
- **SSH:** not required until M4
