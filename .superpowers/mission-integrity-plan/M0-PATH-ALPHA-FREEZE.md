# Path Alpha identity freeze

**Recommendation:** Proceed with the already-measured MLX Qwen2.5 exact-source path. Evidence is complete for identity freeze (historical `measured-run-pass`); not blocked.

**Freeze date:** 2026-08-11  
**Freeze authority packet:** `docs/operations/evidence/2026-08-05-qwen2-mlx-lm-exact-source-refresh/`  
**Baseline packet (preserved, not transferred):** `docs/operations/evidence/2026-08-05-qwen2-mlx-lm-acceptance/`

## Identity table

| Field | Value |
| --- | --- |
| Path ID | `path-alpha-mlx-qlora-v1` |
| Training runtime | `mlx-lm` |
| Method | `qlora` |
| Placement | `single` |
| Model repo | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` |
| Immutable revision | `53a32aee5e9447773fd2b85988395066aef3700a` |
| Policy identity | `model.qwen2-24l.mlx-qlora` version `1.0.0` (configuration-footprint policy, not an artifact allowlist) |
| Policy path | `mlx-lm.qlora.single.dense-causal-lm.v1` |
| Dataset path | `examples/support-sft.jsonl` |
| Dataset SHA-256 | `bf2dca3d6398d639f47a883203920e1f52b0981becac96734147054e53f8aa44` |
| Host class (from evidence) | Apple M5 Pro, arm64, 64 GiB unified memory, macOS 26.6 build 25G72 |
| Python / MLX / MLX-LM pins | Python 3.12.13; MLX 0.31.2; MLX-LM 0.31.3; MLX Metal 0.31.2 |
| Historical acceptance source commit | `719255153e3fc7e38e83b5ff826d587e5e58bf80` (source tree `be99f5664ccb580f2600471f1ae3241a294b1a7e`) |
| Historical bundle fingerprint | `ca2548cf8469fb9867f1558428803b1c9f7c19f48cba754fdb602643f23d1919` |
| Success state | `measured-run-pass` |
| Evidence packet path(s) | `docs/operations/evidence/2026-08-05-qwen2-mlx-lm-exact-source-refresh/`; baseline (unchanged, older fingerprint only): `docs/operations/evidence/2026-08-05-qwen2-mlx-lm-acceptance/` |

### Bound companion identities (same freeze packet; not separate Alpha paths)

| Companion | Value | Source |
| --- | --- | --- |
| Policy snapshot SHA-256 | `c2ae989c8b68df6e984dc7c8670397e791ff30e1f5ce82129e25c1c2b93268d8` | exact-source refresh README / `acceptance-summary.json` |
| Plan ID | `plan_bd33e8ab37765b7efa93` | same |
| Candidate ID | `cand_ee8a9d7fc575e9eeb71b` | same |
| Embedded plan SHA-256 | `673495afd38aee3ebab44c3692fc052bdad5bb0f2c2e8f23fad11e12278dfd23` | same |
| Bundle ZIP SHA-256 | `fcad829b4c845c6b5d1e548b293ec1107ccd7a78ea08b63bc7a1b8ca487be9b1` | same |
| Model weights bytes | `278064920` | same |
| Compiler ID | `mlx-lm.qlora.v1` | `acceptance-summary.json` |
| Plan schema / bundle schema | `aptus.training-plan.v5` / `aptus.bundle.v3` | `acceptance-summary.json` |
| Baseline acceptance source (pre-refresh) | `14ed44b52a76bb84d8d9db4f2303951aa641339b` | baseline packet README |
| Baseline bundle fingerprint (superseded for freeze identity) | `f1d175193792e2b09c606f92c8db1d58e0a7c4bcb531c03c76fc71ad2be10b9e` | baseline packet README |

## Explicit non-claims

- **Not all Qwen2:** scope is the exact pinned `Qwen2.5-0.5B-Instruct-4bit` revision and the reviewed 24-layer dense 4-bit configuration footprint; another matching Qwen2 artifact must complete its own model-data, measured-preflight, and pilot gates.
- **Not CUDA:** no CUDA acceptance is transferred or implied.
- **Not multi-GPU:** placement is single only.
- **Not quality:** training loss, adapter deltas, and reload token generation are structural/runtime evidence, not model quality or safety claims.
- **Not current-HEAD re-proof yet:** historical packet freezes Path Alpha identity; Phase M3 must re-prove at then-current HEAD (or record an explicit, evidence-bound delta). Historical success is not automatic M3 exit.
- **Not production or release readiness** for the whole product; packet is path-scoped exact-source evidence only.
- **Not performance or throughput** guarantees.
- **Not general MLX-LM or Qwen2.5 family certification.**
- **Not resumable checkpoint / full-training resume** claims (MLX artifacts in scope remain weight snapshots where applicable under product rules).
- **Not transfer of the baseline fingerprint** `f1d175…` onto the refreshed fingerprint `ca2548…`; old runtime results do not qualify a later bundle identity.

## Program implication

**M3 must re-prove Path Alpha at then-current HEAD.** This freeze document binds the exact historical identity and measured success state used to stop identity drift. It is the **identity freeze source**, not an automatic M3 exit, not a release gate closeout, and not a claim that current tree still produces the same bundle fingerprint or runtime outcome.

## Sources

Files read for this freeze:

1. `.superpowers/sdd/2026-08-11-mission-trust-when-it-says-no/task-M0.1-brief.md`
2. `docs/operations/evidence/2026-08-05-qwen2-mlx-lm-exact-source-refresh/README.md`
3. `docs/operations/evidence/2026-08-05-qwen2-mlx-lm-exact-source-refresh/acceptance-summary.json`
4. `docs/operations/evidence/2026-08-05-qwen2-mlx-lm-acceptance/README.md`
5. `docs/operations/evidence/2026-08-05-qwen2-mlx-lm-acceptance/model-policy-snapshot.v1.json`
6. `docs/operations/evidence/2026-08-05-qwen2-mlx-lm-acceptance/runtime-environment.json`
7. `docs/product/mission-integrity-plan.md` (M0.2 table shape / M3 re-proof expectation)
8. `.superpowers/sdd/2026-08-11-mission-trust-when-it-says-no/ (pointer retired; authority: docs/product/mission-integrity-plan.md)`
9. `.superpowers/mission-integrity-plan/README.md`

## Self-check

- [x] No invented revisions, commits, fingerprints, or digests: every hash and revision above appears in a cited evidence file.
- [x] Immutable model revision `53a32aee5e9447773fd2b85988395066aef3700a` appears in exact-source refresh README and `acceptance-summary.json`.
- [x] Dataset SHA-256 `bf2dca3d6398d639f47a883203920e1f52b0981becac96734147054e53f8aa44` appears in the same.
- [x] Historical acceptance source commit `719255153e3fc7e38e83b5ff826d587e5e58bf80` and bundle fingerprint `ca2548cf8469fb9867f1558428803b1c9f7c19f48cba754fdb602643f23d1919` appear in the exact-source refresh packet.
- [x] Baseline commit `14ed44b52a76bb84d8d9db4f2303951aa641339b` and baseline fingerprint `f1d175193792e2b09c606f92c8db1d58e0a7c4bcb531c03c76fc71ad2be10b9e` appear only as historical baseline companions, not as the freeze primary bundle identity.
- [x] Success state is historical `measured-run-pass` only; M3 not marked complete.
- [x] Single Alpha model/path chosen; no second Alpha model.
