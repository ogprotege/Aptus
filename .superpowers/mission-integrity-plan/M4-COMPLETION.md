# M4 — COMPLETION NOTE

> **Phase status:** **COMPLETE** (awaiting PR merge)

| Field | Value |
| --- | --- |
| Phase | M4 |
| Title | Path Beta release-honest CUDA LoRA single-device handoff |
| Host | Ubuntu 24.04.4 LTS, NVIDIA GeForce RTX 3050, driver 595.84 (logical host: Sherminator) |
| Source base at measurement | `93d69f63c7d3c1147ce186e810c355cdcf1a1b9c` |
| Product fix | CUDA PEP 440 public-version dependency pin match (`cuda/preflight.py`) |
| Clean ladders | **1** × `measured-run-pass` |
| Artifact fingerprint | `1a41e586511cff2cf68b1e0794a9b1b57395601a072fc4661bf0ebff140bf855` |
| Plan / candidate | `plan_6870eaf879c843dd0ede` / `cand_2fe2c0a05360293358f6` |
| Adapter SHA-256 | `5aab8b259824a1dc81613c01e6ea49cb2d757e9601d5f080c009aabef9eafffa` |
| Semantic adapter reload | **Not claimed** |

## Deliverables

- `docs/guides/path-beta-cuda-lora-operator.md`
- `docs/operations/evidence/2026-08-12-path-beta-cuda-lora-m4/`
- Capability + inventory + operations index sync
- Unit test: `test_cuda_dependency_pins_accept_pep440_local_labels`

## Mission check

- Exact Path Beta identity: **yes**
- Clean-env dependency install: **yes** (`torch 2.13.0+cu130`)
- Five-job ordered ladder: **yes**
- Structural export + parent `verified-at-completion`: **yes**
- Claim boundary non-transfers: **yes**
- Job-control cancel smoke (M4.4): **pass** (`job_70d112eb…` → `cancelled`, rc `-15`, lease reconciled; see evidence `m4.4-cancel-smoke.json`)

## Next

- Open/merge PR for M4
- Optional M4.4 job-control cancel smoke on host
- Program next: M5 KISS correction loop (design notes only until evidence green)
