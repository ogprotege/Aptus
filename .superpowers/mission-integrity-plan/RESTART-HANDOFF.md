# Restart handoff — 2026-08-12

> User restarting machine. Read this first on resume.

## Where we got to

**Mission Integrity Program: M0 → M3 complete and on `main`.**

| Phase | What | GitHub | Result |
| --- | --- | --- | --- |
| M0 | Mission freeze (Path Alpha/Beta identities) | PR #85 (+ plan #84) | Done |
| M1 | Promise audit / gap register / false-yes | PR #85 | Done |
| M2 | Structured plan refusal guidance | PR #86 | Done (`src/aptus/refusal.py`, `web/src/lib/refusal.ts`) |
| **M3** | **Path Alpha MLX release-honest acceptance** | **PR #87 MERGED** | **Done** |
| M4 | Path Beta CUDA | — | **Not started** |

### Main tip (after merge)

- Merge commit: `93d69f63c7d3c1147ce186e810c355cdcf1a1b9c`
- PR: https://github.com/ogprotege/Aptus/pull/87  
- Title: `docs: Path Alpha MLX measured acceptance (M3)`
- Evidence commit: `a8b0751`
- Measurement source (before M3 docs commit): `f4775c01…` (post-M2)

### After restart — sync local

```bash
cd /Users/biscuit/Aptus
git fetch origin
git checkout main
git pull origin main
# expect tip: 93d69f6 Merge pull request #87 ...
```

## M3 substance (do not re-run unless pins change)

- **Runbook:** `docs/guides/path-alpha-mlx-operator.md`
- **Evidence:** `docs/operations/evidence/2026-08-12-path-alpha-mlx-m3/`
- **Two** clean ladders → `measured-run-pass` (3 optimizer updates each)
- **Identity:** `mlx-community/Qwen2.5-0.5B-Instruct-4bit` @ `53a32aee…`
- **Dataset:** `examples/support-sft.jsonl`
- **Runtime:** mlx 0.31.2 / mlx-lm 0.31.3
- **Fingerprint:** `ace50ce8…` (fresh; does **not** transfer historical `ca2548cf…`)
- **Adapter SHA (both runs):** `4717543bb38f0845…` (matches historical acceptance adapter)
- **Host:** Apple M5 Pro, 64 GiB, macOS 26.6.1
- **Not done in M3:** M3.6 full workbench UI walk (CLI/managed only)
- **Doc lag:** plan §10 checkboxes in `docs/product/mission-integrity-plan.md` may still show `- [ ]`

## Machine cleanup done this session

| Action | Outcome |
| --- | --- |
| Delete `.aptus-m3-work`, `.aptus-m3-state*` | ~689 M |
| Delete `~/.cache/huggingface` | ~1.5 G |
| Delete **all 22** CoreSimulator devices | **17 G → 4 K** |
| Left alone | Claude AS (~8.6 G), rustup (5.5 G), `~/.cache` codex/uv/puppeteer (~2.9 G) |

M3 Git evidence was **not** deleted — only rebuildable runtime caches and sims.

## Next (when ready)

1. `git pull` main as above.  
2. **Default next phase: M4 Path Beta (CUDA)** — only with explicit host/SSH + cost authorization.  
3. Optional later: M3.6 workbench UI walk; tick mission-plan §10 checkboxes to match merged reality.  
4. Do **not** re-prove Path Alpha unless MLX pins or Alpha identity change.

## Workspace notes

- Program plan: `docs/product/mission-integrity-plan.md`
- Local phase notes: `.superpowers/mission-integrity-plan/` (`M0…M3-COMPLETION.md`, `STATUS.md`)
- Mnemoverse domain `project:aptus` also has this handoff (query: “mission integrity M3 handoff”).
