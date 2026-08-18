# Lane 3 — Run disposition specification (freeze)

> **Status:** APPROVED 2026-08-18 — owner replied "Approved"  
> **Authority:** Subordinate to claim language, current capabilities, and mission invariants I1–I12  
> **Increment:** Lane 3. **Not M10.** **Not a 0.2 ship.**  
> **Implementation plan:** `docs/superpowers/plans/2026-08-18-lane-3-run-disposition.md`  
> **Last reviewed:** 2026-08-18  
> **Next scheduled review:** After Tasks 1–6 merge, or before writing docs/product/0.2-cut-note.md

Owner sign-off (chat 2026-08-18: "Approved"):

- Approach **A** (presentation object + human 0.2 cut note)
- Verbs **use / done / stop** (not cut — `cut` is reserved for the Aptus 0.2 product note)
- Attach as a per-job last call; `spec-plan` stays open
- Responsibility door: Aptus names caution; the operator attests and takes responsibility

---

## 1. Goal

After a completed train, Aptus already has a parent validation state, a
training-signal `run_correction`, and an optional gold `evaluation-result`.
None of those say what to do with the result.

Lane 3 adds one last call: **use**, **done**, or **stop**. The operator attests
it. Aptus never infers the kind from a parent pass, a loss series, or a gold
score.

A second, later artifact — `docs/product/0.2-cut-note.md` — is a human note
about whether Aptus 0.2 is keep-building / cut-freeze / stop. Journey A and
Journey B dispositions are evidence for that note, not a quality grade.

## 2. Non-goals

- Inferring Use from `measured-run-pass`
- Inferring Stop from gold exact-match `fail` or score `0.0`
- AutoML, a third epoch, or rank/LR search
- Relabeling 7B as Path Alpha or a reviewed identity
- Making `spec-plan`, compile, or the ladder infeasible because of a disposition
- Revoking `measured-run-pass`
- Unsloth-style Train chrome
- Committing `aptus-work/`
- Naming this increment M10
- A second schema for the 0.2 product cut in this increment

## 3. Responsibility door

Same philosophy as `--confirm-unreviewed-runtime` and `--confirm-full-train`.

Aptus stays the referee: it shows the stacked evidence and will not turn a
caution into a silent yes. The operator overrides by **explicitly** choosing
`use`, `done`, or `stop` and thereby takes responsibility for that last call.

- `aptus dispose --kind …` is that attest. There is no default kind.
- Changing your mind is another explicit `dispose` (last attest wins).
- `spec-plan` / compile / the ladder remain open. Aptus will not hall-monitor a
  later plan. It also will not pretend the prior last call never happened:
  `jobs` still shows it.
- A later increment may add `--confirm-reopen-disposition` if we want a lock
  with a door. **Not this cut.**

## 4. Meanings (novice and expert, same act)

Ask one question: **What do you want to do with what you just trained?**

| Kind | Novice | Expert |
| --- | --- | --- |
| `use` | I’ll load this and try it. | Consume the adapter. Line stays open. Not quality. |
| `done` | I’m finished training this. Leave the files. Don’t start another train on the same recipe. | Close the identity. Retain the artifact. No suggested next-plan. Not a 0.2 ship. |
| `stop` | Don’t use this. Don’t train this again. | Do not consume. Do not suggest a replan. Parent pass is not revoked. |

`cut` is **not** a run-disposition kind. It names the product note in §12.

## 5. When it exists

Attaches only to a **completed `train` job**.

Missing file / missing field means **no last call recorded**, not `use`.

Pilot, preflight, model-data, and dependency jobs cannot receive a
disposition.

## 6. Persistence

Sibling of the job record, same state directory and privacy:

```text
{state-dir}/jobs/{job_id}.disposition.json
```

- Mode `0600`, no symlink, no-clobber of a different `job_id`.
- Job GET and `aptus jobs --id` attach `run_disposition` at read time.
- Do not add `run_disposition` to `aptus.job-record.v1` identity.
- Do not write into the bundle or `validation-report.json`.

Last attest overwrites the file and may keep `previous_kind`.

## 7. Schema `aptus.run-disposition.v1`

```json
{
  "schema_version": "aptus.run-disposition.v1",
  "kind": "use | done | stop",
  "job_id": "job_…",
  "plan_id": "plan_…",
  "candidate_id": "cand_…",
  "run_id": "run_… | null",
  "attested_at": "ISO-8601",
  "previous_kind": "use | done | stop | null",
  "source": "operator-attested",
  "evidence": {
    "validation_state": "measured-run-pass | …",
    "run_correction_kind": "none | loss-flat | loss-collapsed | eval-rose | null",
    "evaluation_decision": "pass | fail | abstain | omitted"
  },
  "operator_next_step": {
    "action": "load-adapter | none",
    "label": "string"
  },
  "non_claims": [
    "Training finished is not this decision.",
    "Training loss is not this decision.",
    "Gold exact-match is not general model quality.",
    "This is not a 0.2 ship, freeze, or stop."
  ]
}
```

`evidence` is a snapshot of what Aptus showed when the operator attested. It is
not a proof that those facts still hold. It is not hashed into `plan_id`.

`operator_next_step.action` is `load-adapter` only for `use`; otherwise `none`.

Required `non_claims` must all be present (same rule as evaluation contracts).

## 8. CLI and API

```bash
aptus dispose JOB_ID --kind {use,done,stop} --state-dir STATE
```

Refuse unless the job exists, `action` is `train`, and `state` is `completed`.

`aptus jobs --id` prints a stderr block after run-correction:

```text
Aptus run disposition (operator-attested; not quality):
  kind: done
  next: none — I'm finished training this.
```

Job JSON includes `run_disposition` when the sibling file is valid. Corrupt or
symlinked sibling → quarantine/reason, do not invent `use`.

OpenAPI: optional `run_disposition` on the job response. Regenerate; do not
hand-edit.

## 9. Workbench

Run stage, only after a completed train:

- Read-only stacked evidence: parent state, `run_correction.kind`, optional
  eval decision.
- Three choices: Use it / I'm done training this / Don't use it.
- After attest, the primary next label follows the disposition, not
  `replan-with-more-epochs`, when kind is `done` or `stop`.
- Eyebrow: operator-attested. Never “recommended” or “quality pass.”

## 10. Identity and status machine

- Do not put disposition into `plan_id` or `candidate_id`.
- Do not bump memory formula versions.
- Do not change candidate `feasible` / `conditional` / `infeasible` /
  `unsupported` from a disposition.
- Do not change `measured-run-pass` rules.

## 11. Claim sentences

Use:

- “operator-attested run disposition”;
- “no last call recorded”;
- “Use is not a quality yes”;
- “Done closes this recipe; it does not ship Aptus 0.2.”

Do not use:

- “Aptus decided this adapter is good”;
- “gold fail means Stop”;
- “measured-run-pass means Use”;
- “cut this run” (wrong verb);
- “reviewed 7B” after Use on an unreviewed runtime.

## 12. Layer 2 — 0.2 cut note (same increment, after layer 1 ships)

Write `docs/product/0.2-cut-note.md` only after Journey A and Journey B can
show a recorded disposition (or an explicit “no last call”).

The note chooses one of: keep building 0.2 / cut-freeze 0.2 as the five-stage
referee / stop the 0.2 line. It cites operational evidence. It does not cite
loss or gold as quality.

No `aptus.product-cut.v1` schema in this increment.

## 13. Verification (when implemented)

- Unittest: refuse dispose on non-train / non-completed; attach on job GET;
  missing sibling is absent not `use`; last attest wins; `plan_id` unchanged.
- CLI + OpenAPI + Run stage copy.
- Claim-language and CLI docs in the same change.
- No CUDA/MLX pilot required (no trainer, memory, or export change).

## 14. Frozen decisions

| Decision | Choice |
| --- | --- |
| Approach | A |
| Kinds | `use`, `done`, `stop` |
| Who writes | operator attest |
| Default kind | none (absent) |
| Blocks planner | no |
| Blocks ladder | no |
| Revokes parent pass | no |
| Product “cut 0.2” | layer-2 note, later in this increment |
