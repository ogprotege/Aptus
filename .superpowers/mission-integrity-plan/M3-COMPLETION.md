# M3 — COMPLETION NOTE

> **Phase status:** **COMPLETE** (pending PR merge)

| Field | Value |
| --- | --- |
| Phase | M3 |
| Title | Path Alpha release-honest MLX QLoRA |
| Host | Apple M5 Pro, 64 GiB, macOS 26.6.1 (25G76) |
| Source at measurement | `f4775c01e6b8f932e11c2d665e90859d6aedbe04` |
| Clean ladders | **2** × `measured-run-pass` |
| Artifact fingerprint | `ace50ce8b4defc2a3a871e4031a358e0942fb114980e487acac07c66f766ce14` |
| Adapter SHA-256 | `4717543bb38f084573a6f1ea2fa0638d71c1a1a38b1b2103545951e052d5f31b` (both runs; matches historical) |

## Deliverables

- `docs/guides/path-alpha-mlx-operator.md`
- `docs/operations/evidence/2026-08-12-path-alpha-mlx-m3/`
- Capability + inventory sync

## Mission check

- Two clean measured-run-pass ladders: **yes**
- Exact Path Alpha identity: **yes**
- Claim boundary non-transfers: **yes**
- Second-run offline-only attempt failed (HF offline API list); retry with warm cache online list succeeded — documented as operator note

## Next

- **M4** Path Beta CUDA (SSH/host) — not started
