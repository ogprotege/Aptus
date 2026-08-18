# Lane 3 Run Disposition Implementation Plan

> **Status:** Active | **Authority:** Implementation plan (subordinate to claim language and Lane 3 spec freeze) | **Applies to:** Aptus Lane 3 run-disposition increment, not M10, not a 0.2 ship | **Audience:** Agents executing Tasks 1–6 | **Last reviewed:** 2026-08-18 | **Review by:** After Tasks 1–6 merge, or before Task 7

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After a completed train, let the operator attest Use / Done / Stop as a last call without turning `measured-run-pass`, loss, or gold into a quality yes.

**Architecture:** Add presentation-only `aptus.run-disposition.v1`, persisted as `{state-dir}/jobs/{job_id}.disposition.json`. `JobService.get` attaches it the same way it attaches `run_correction`. CLI `aptus dispose` and `POST /api/v1/jobs/{job_id}/disposition` are the responsibility door. The planner status machine, `plan_id`, and `measured-run-pass` do not change. Layer 2 (`docs/product/0.2-cut-note.md`) waits until Journey A and B have a recorded disposition.

**Tech Stack:** Python 3.11+ unittest (not pytest), `src/aptus/` JobService + CLI + FastAPI, generated OpenAPI, React Run stage, claim-language docs.

## Global Constraints

- Spec freeze: `docs/superpowers/specs/2026-08-18-lane-3-run-disposition-design.md`
- Increment name is Lane 3. Not M10. Not a 0.2 ship.
- Kinds are only `use`, `done`, `stop`. Never `cut` on a run.
- Aptus never infers the kind. Missing sibling means no last call, not `use`.
- Do not put disposition in `plan_id` / `candidate_id`.
- Do not bump `aptus-memory-v2` / `aptus-memory-mlx-v2`.
- Do not change candidate status or `measured-run-pass` from a disposition.
- `spec-plan`, compile, and the ladder stay open. No `--confirm-reopen-disposition` in this cut.
- Last attest wins; keep `previous_kind`.
- Unittest only: `PYTHONPATH=src:. python -m unittest …`
- Never hand-edit `docs/reference/openapi.v1.json` or `web/src/generated/openapi.ts`.
- Update claim language in the same change as behavior.
- No CUDA/MLX pilot (no trainer, memory, or export change).
- Do not commit `aptus-work/`.

---

## File map

| File | Responsibility |
| --- | --- |
| `src/aptus/run_disposition.py` | Pure build/parse of `aptus.run-disposition.v1`. No I/O. |
| `src/aptus/execution.py` | `JobDispositionError`; write/read sibling; attach on `get`. |
| `src/aptus/cli.py` | `dispose` command; jobs stderr block. |
| `src/aptus/api.py` / `api_contracts.py` | `JobResponse.run_disposition`; POST disposition. |
| `web/src/types.ts`, `web/src/api.ts` | Types, path, `disposeJob`, normalizer. |
| `web/src/components/RunConsole.tsx` | Evidence + three choices. |
| `web/src/stages/RunStage.tsx`, `web/src/App.tsx` | Wire `onDisposeJob`. |
| `tests/aptus/test_run_disposition.py` | Pure schema tests. |
| `tests/aptus/test_execution.py` (or new `test_job_disposition.py`) | Persist/attach/refuse. |
| `tests/aptus/test_cli.py` | dispose + jobs stderr. |
| `tests/aptus/test_api.py` | POST + GET field. |
| `web/src/stages/RunStage.test.tsx` | Copy and buttons. |
| `docs/reference/cli.md` | Parser JSON + prose. |
| `docs/reference/api.md`, `docs/reference/error-codes.md` | Route + `job_disposition_refused`. |
| `docs/product/claim-language.md` | Allowed/forbidden sentences. |
| `README.md` | Command table row. |
| `docs/product/0.2-cut-note.md` | Layer 2 only after A/B dispositions exist. |

Do **not** modify: `planning.py` status rules, `plan_contract.py` identity, bundle `train.py`, memory formulas, M8 eval schemas, Path Alpha freeze.

---

### Task 1: Pure disposition object

**Files:**
- Create: `src/aptus/run_disposition.py`
- Test: `tests/aptus/test_run_disposition.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `DISPOSITION_SCHEMA_VERSION = "aptus.run-disposition.v1"`
  - `DISPOSITION_KINDS = frozenset({"use", "done", "stop"})`
  - `DISPOSITION_NON_CLAIMS` = the four spec sentences, exact
  - `RunDisposition` dataclass with `to_primitive() -> dict[str, object]`
  - `build_run_disposition(*, kind: str, job_id: str, plan_id: str, candidate_id: str, run_id: str | None, attested_at: str, previous_kind: str | None, validation_state: str | None, run_correction_kind: str | None, evaluation_decision: str | None) -> RunDisposition`
  - `run_disposition_from_primitive(payload: Mapping[str, Any]) -> RunDisposition`
  - `next_step_for_kind(kind: str) -> tuple[str, str]` → `("load-adapter", "Load this adapter")` for `use`; `("none", "I'm finished training this.")` for `done`; `("none", "Don't use this. Don't train this again.")` for `stop`

- [ ] **Step 1: Write the failing tests**

```python
# tests/aptus/test_run_disposition.py
import unittest

from aptus.run_disposition import (
    DISPOSITION_NON_CLAIMS,
    build_run_disposition,
    run_disposition_from_primitive,
)


class RunDispositionTests(unittest.TestCase):
    def test_use_sets_load_adapter_and_required_non_claims(self) -> None:
        body = build_run_disposition(
            kind="use",
            job_id="job_" + "a" * 32,
            plan_id="plan_abc",
            candidate_id="cand_abc",
            run_id="run_abc",
            attested_at="2026-08-18T00:00:00+00:00",
            previous_kind=None,
            validation_state="measured-run-pass",
            run_correction_kind="none",
            evaluation_decision="omitted",
        ).to_primitive()
        self.assertEqual(body["schema_version"], "aptus.run-disposition.v1")
        self.assertEqual(body["kind"], "use")
        self.assertEqual(body["source"], "operator-attested")
        self.assertEqual(body["operator_next_step"]["action"], "load-adapter")
        self.assertEqual(body["operator_next_step"]["label"], "Load this adapter")
        for claim in DISPOSITION_NON_CLAIMS:
            self.assertIn(claim, body["non_claims"])

    def test_done_and_stop_have_no_next_plan(self) -> None:
        done = build_run_disposition(
            kind="done",
            job_id="job_" + "b" * 32,
            plan_id="plan_abc",
            candidate_id="cand_abc",
            run_id=None,
            attested_at="2026-08-18T00:00:00+00:00",
            previous_kind="use",
            validation_state="measured-run-pass",
            run_correction_kind="loss-flat",
            evaluation_decision="fail",
        )
        self.assertEqual(done.operator_next_step.action, "none")
        self.assertEqual(done.previous_kind, "use")
        stop = build_run_disposition(
            kind="stop",
            job_id="job_" + "c" * 32,
            plan_id="plan_abc",
            candidate_id="cand_abc",
            run_id=None,
            attested_at="2026-08-18T00:00:00+00:00",
            previous_kind=None,
            validation_state="measured-run-pass",
            run_correction_kind=None,
            evaluation_decision="omitted",
        )
        self.assertEqual(stop.operator_next_step.label, "Don't use this. Don't train this again.")

    def test_unknown_kind_and_cut_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_run_disposition(
                kind="cut",
                job_id="job_" + "d" * 32,
                plan_id="plan_abc",
                candidate_id="cand_abc",
                run_id=None,
                attested_at="2026-08-18T00:00:00+00:00",
                previous_kind=None,
                validation_state=None,
                run_correction_kind=None,
                evaluation_decision="omitted",
            )

    def test_from_primitive_requires_all_non_claims(self) -> None:
        payload = build_run_disposition(
            kind="use",
            job_id="job_" + "e" * 32,
            plan_id="plan_abc",
            candidate_id="cand_abc",
            run_id=None,
            attested_at="2026-08-18T00:00:00+00:00",
            previous_kind=None,
            validation_state=None,
            run_correction_kind=None,
            evaluation_decision="omitted",
        ).to_primitive()
        payload["non_claims"] = list(payload["non_claims"])[:-1]
        with self.assertRaises(ValueError):
            run_disposition_from_primitive(payload)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src:. python -m unittest tests.aptus.test_run_disposition -v`

Expected: FAIL with `ModuleNotFoundError: aptus.run_disposition`

- [ ] **Step 3: Write minimal implementation**

Create `src/aptus/run_disposition.py` mirroring `src/aptus/training_policy.py` dataclasses. Reject `kind` not in `{use,done,stop}`. `evaluation_decision` allowed values: `pass`, `fail`, `abstain`, `omitted`. `from_primitive` requires `schema_version`, `source == "operator-attested"`, and every `DISPOSITION_NON_CLAIMS` member.

- [ ] **Step 4: Run tests and make sure they pass**

Run: `PYTHONPATH=src:. python -m unittest tests.aptus.test_run_disposition -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/aptus/run_disposition.py tests/aptus/test_run_disposition.py
git commit -m "feat: add aptus.run-disposition.v1 object"
```

---

### Task 2: Persist and attach on the job

**Files:**
- Modify: `src/aptus/execution.py` (`JobPrerequisiteError` neighborhood ~125; `JobService._record_path` ~2758; `get` ~4494; `_attach_run_correction` ~4645)
- Test: `tests/aptus/test_job_disposition.py`

**Interfaces:**
- Consumes: `build_run_disposition`, `run_disposition_from_primitive` from Task 1
- Produces:
  - `class JobDispositionError(ValueError)` with `code = "job_disposition_refused"`
  - `JobService._disposition_path(self, job_id: str) -> Path` → `self.root / f"{job_id}.disposition.json"`
  - `JobService.save_disposition(self, job_id: str, kind: str) -> dict[str, Any]`
  - `JobService._attach_run_disposition(self, record: dict[str, Any]) -> None`

`save_disposition` must `get(job_id)` first, refuse unless `action == "train"` and `state == "completed"`, snapshot `validation_report.state`, `run_correction.kind`, and `evaluation_decision="omitted"` unless the record already has an evaluation decision, write mode `0600` atomically, set `previous_kind` from an existing valid sibling. Return `get(job_id)` with `run_disposition` attached.

`_attach_run_disposition`: if sibling missing, do nothing. If symlink or unreadable or `from_primitive` fails, set `record["run_disposition_error"]` and do **not** set `run_disposition`. Never invent `use`.

Call `_attach_run_disposition` from `get` immediately after `_attach_run_correction`.

- [ ] **Step 1: Write the failing tests**

Use a temp `JobService` root. Prefer constructing a completed train record the same way existing execution tests write job JSON (copy the smallest helper already used for completed-train fixtures in `tests/aptus/test_execution.py`). Required cases:

```python
def test_save_disposition_refuses_non_train_and_non_completed(self) -> None: ...
def test_save_disposition_writes_sibling_and_get_attaches(self) -> None: ...
def test_second_attest_sets_previous_kind(self) -> None: ...
def test_missing_sibling_is_absent_not_use(self) -> None: ...
def test_corrupt_sibling_sets_error_not_use(self) -> None: ...
```

Also assert `save_disposition` does not rewrite `plan_id` on the job record.

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src:. python -m unittest tests.aptus.test_job_disposition -v`

Expected: FAIL (`JobDispositionError` / `save_disposition` missing)

- [ ] **Step 3: Write minimal implementation**

Add the error class and the three methods. Follow `_write` atomic replace. `os.chmod(path, 0o600)`.

- [ ] **Step 4: Run tests and make sure they pass**

Run: `PYTHONPATH=src:. python -m unittest tests.aptus.test_job_disposition tests.aptus.test_run_disposition -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/aptus/execution.py tests/aptus/test_job_disposition.py
git commit -m "feat: persist operator-attested run disposition beside job records"
```

---

### Task 3: CLI `dispose` and `jobs` presentation

**Files:**
- Modify: `src/aptus/cli.py` (parser ~576; `jobs` handler ~1072; print helpers ~136)
- Modify: `docs/reference/cli.md` (summary table, `<!-- aptus-cli-parser-contract -->` JSON, new prose section after `aptus jobs`)
- Modify: `README.md` command table (~219)
- Test: `tests/aptus/test_cli.py`

**Interfaces:**
- Consumes: `JobService.save_disposition`, `JobDispositionError`
- Produces: `aptus dispose JOB_ID --kind {use,done,stop} --state-dir STATE`

Parser:

```python
dispose = commands.add_parser(
    "dispose",
    help="Attest Use, Done, or Stop for a completed train job (not quality).",
)
dispose.add_argument("job_id")
dispose.add_argument("--kind", required=True, choices=("use", "done", "stop"))
dispose.add_argument("--state-dir", type=Path, default=Path(".aptus-state"))
```

Handler: `JobService(arguments.state_dir / "jobs").save_disposition(...)`; print disposition stderr block; write job JSON to stdout. Map `JobDispositionError` and `KeyError` to `Aptus error:` exit 2.

Jobs `--id`: after `_print_job_run_correction`, call `_print_job_run_disposition`.

```python
def _emit_run_disposition_block(payload: Mapping[str, Any]) -> None:
    print(
        "\n".join(
            [
                "Aptus run disposition (operator-attested; not quality):",
                f"  kind: {payload.get('kind')}",
                f"  next: {payload.get('operator_next_step', {}).get('action')} — "
                f"{payload.get('operator_next_step', {}).get('label')}",
            ]
        ),
        file=sys.stderr,
    )
```

CLI parser JSON must add:

```json
"aptus dispose": {
  "<job_id>": {"default": null},
  "--kind": {"choices": ["use", "done", "stop"], "default": null},
  "--state-dir": {"default": ".aptus-state"}
}
```

- [ ] **Step 1: Write the failing CLI tests**

Mirror `test_jobs_id_prints_training_signal_correction_block`:

- `test_dispose_writes_disposition_and_prints_block`
- `test_dispose_refuses_without_kind`
- `test_jobs_id_prints_disposition_block`

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src:. python -m unittest tests.aptus.test_cli.CliTests.test_dispose_writes_disposition_and_prints_block tests.aptus.test_documentation.DocumentationTests.test_cli_reference_matches_parser_choices_and_defaults -v`

Expected: FAIL (unknown command `dispose` and/or parser contract drift)

- [ ] **Step 3: Implement CLI + docs**

- [ ] **Step 4: Run tests**

Run:

```bash
PYTHONPATH=src:. python -m unittest tests.aptus.test_cli tests.aptus.test_documentation.DocumentationTests.test_cli_reference_matches_parser_choices_and_defaults -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/aptus/cli.py docs/reference/cli.md README.md tests/aptus/test_cli.py
git commit -m "feat: add aptus dispose responsibility door"
```

---

### Task 4: HTTP contract

**Files:**
- Modify: `src/aptus/api_contracts.py` (`JobResponse` ~991)
- Modify: `src/aptus/api.py` (jobs routes ~2030)
- Modify: `docs/reference/api.md`, `docs/reference/error-codes.md`
- Generate: `docs/reference/openapi.v1.json`, `web/src/generated/openapi.ts`
- Test: `tests/aptus/test_api.py`

**Interfaces:**
- Consumes: `JobService.save_disposition`
- Produces:
  - `JobResponse.run_disposition: RunDispositionResponse | None = None`
  - `class DisposeJobRequest(ClosedResponseModel): kind: Literal["use", "done", "stop"]`
  - `POST /api/v1/jobs/{job_id}/disposition` → `JobResponse`
  - 404 `job_not_found`; 409 `{error: "job_disposition_refused", message}`

```python
class RunDispositionNextStepResponse(ClosedResponseModel):
    action: Literal["load-adapter", "none"]
    label: str

class RunDispositionEvidenceResponse(ClosedResponseModel):
    validation_state: str | None
    run_correction_kind: str | None
    evaluation_decision: Literal["pass", "fail", "abstain", "omitted"]

class RunDispositionResponse(ClosedResponseModel):
    schema_version: Literal["aptus.run-disposition.v1"]
    kind: Literal["use", "done", "stop"]
    job_id: str
    plan_id: str
    candidate_id: str
    run_id: str | None
    attested_at: str
    previous_kind: Literal["use", "done", "stop"] | None
    source: Literal["operator-attested"]
    evidence: RunDispositionEvidenceResponse
    operator_next_step: RunDispositionNextStepResponse
    non_claims: list[str]
```

- [ ] **Step 1: Write failing API tests** (POST on completed train; 409 on pilot; GET includes field after POST; GET without sibling omits field)

- [ ] **Step 2: Run to verify fail**

Run: `PYTHONPATH=src:. python -m unittest tests.aptus.test_api -v` (new methods only if you can select them)

Expected: FAIL (route missing)

- [ ] **Step 3: Implement route + contracts, then regenerate**

```bash
.venv/bin/python tools/generate_openapi.py
npm --prefix web run openapi:generate
.venv/bin/python tools/check_client_contracts.py
```

- [ ] **Step 4: Run API + contract checks**

Expected: PASS

- [ ] **Step 5: Commit** including generated OpenAPI + TS

```bash
git add src/aptus/api.py src/aptus/api_contracts.py docs/reference/api.md docs/reference/error-codes.md docs/reference/openapi.v1.json web/src/generated/openapi.ts tests/aptus/test_api.py
git commit -m "feat: expose run disposition on jobs API"
```

---

### Task 5: Workbench last call

**Files:**
- Modify: `web/src/types.ts` (after `RunCorrection` ~429; `Job` ~834)
- Modify: `web/src/api.ts` (`API_PATHS` ~83; `normalizeJob` ~538; new `disposeJob`)
- Modify: `web/src/components/RunConsole.tsx` (after `RunCorrectionPanel`)
- Modify: `web/src/stages/RunStage.tsx` (new `onDisposeJob`)
- Modify: `web/src/App.tsx` (`handleCreateJob` neighborhood ~836)
- Test: `web/src/stages/RunStage.test.tsx`, `web/src/api.test.ts`

**Interfaces:**
- Consumes: `POST /api/v1/jobs/{job_id}/disposition`
- Produces: `api.disposeJob(id, kind: "use" | "done" | "stop")`

Copy (exact):

- Eyebrow: `Operator-attested`
- Title: `What do you want to do with what you just trained?`
- Buttons: `Use it` / `I'm done training this` / `Don't use it`
- Provenance badge: `kind="user-attested"` label `Operator`
- Show only when `job.mode === "train"` (or `action === "train"`) and `job.state === "completed"`
- After `done` or `stop`, do not present run-correction `replan-with-fact-hints` as the primary next line; show disposition next instead
- Never say recommended, quality pass, or cut

- [ ] **Step 1: Write failing component test** that a completed train job without disposition shows the three buttons and that clicking `I'm done training this` calls `onDisposeJob("done")`.

- [ ] **Step 2: Run** `npm --prefix web test -- src/stages/RunStage.test.tsx`

Expected: FAIL (prop/buttons missing)

- [ ] **Step 3: Implement types, client, panel, App wiring**

- [ ] **Step 4: Run** `npm --prefix web test` and `npm --prefix web run typecheck`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add web/src/types.ts web/src/api.ts web/src/api.test.ts web/src/components/RunConsole.tsx web/src/stages/RunStage.tsx web/src/stages/RunStage.test.tsx web/src/App.tsx
git commit -m "feat: add operator-attested Use/Done/Stop on Run"
```

---

### Task 6: Claim language

**Files:**
- Modify: `docs/product/claim-language.md` (planning/run claims)
- Modify: `docs/product/current-capabilities.md` only if it lists post-run objects (add one sentence: operator-attested run disposition is not quality)

**Interfaces:** none

Add to allowed:

- “operator-attested run disposition”
- “no last call recorded”
- “Use is not a quality yes”
- “Done closes this recipe; it does not ship Aptus 0.2.”

Add to forbidden:

- “Aptus decided this adapter is good”
- “gold fail means Stop”
- “measured-run-pass means Use”
- “cut this run”
- “reviewed 7B” after Use on an unreviewed runtime

- [ ] **Step 1: Add a documentation test** in `tests/aptus/test_documentation.py` that `claim-language.md` contains `operator-attested run disposition` and `Use is not a quality yes`.

- [ ] **Step 2: Run it to see fail** if the sentences are not yet in the file, then add them.

- [ ] **Step 3: Run** `PYTHONPATH=src:. python -m unittest tests.aptus.test_documentation -v`

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add docs/product/claim-language.md docs/product/current-capabilities.md tests/aptus/test_documentation.py
git commit -m "docs: lock Lane 3 disposition claim language"
```

---

### Task 7: Layer 2 0.2 cut note (gated)

**Do not start this task until Tasks 1–6 are merged or otherwise usable, and the owner has recorded a disposition (or an explicit “no last call”) on Journey A `job` in `aptus-work/test-drive-state` and Journey B `job_87419924d0f941a580d83979fb9b0f9f` in `aptus-work/magisterium-state`.**

**Files:**
- Create: `docs/product/0.2-cut-note.md`

The note chooses exactly one of: keep building 0.2 / cut-freeze 0.2 as the five-stage referee / stop the 0.2 line. Cite operational evidence only (`measured-run-pass`, dispositions). Do not cite loss or gold as quality. No `aptus.product-cut.v1` schema.

- [ ] **Step 1: Confirm both jobs have `run_disposition` or a written “no last call” from the owner**
- [ ] **Step 2: Draft the note and ask the owner which of the three product choices it records**
- [ ] **Step 3: Commit only after the owner picks the product choice**

```bash
git add docs/product/0.2-cut-note.md
git commit -m "docs: record the Aptus 0.2 cut note"
```

---

## Spec coverage

| Spec § | Task |
| --- | --- |
| 1 Goal | all |
| 2 Non-goals | Global Constraints + Tasks 2/7 |
| 3 Responsibility door | Tasks 3–5 |
| 4 Meanings / verbs | Task 1 labels |
| 5 When it exists | Task 2 refuse rules |
| 6 Persistence | Task 2 |
| 7 Schema | Task 1 |
| 8 CLI/API | Tasks 3–4 |
| 9 Workbench | Task 5 |
| 10 Identity | Task 2 plan_id test; no planning.py edits |
| 11 Claim sentences | Task 6 |
| 12 Layer 2 | Task 7 (gated) |
| 13 Verification | each task’s tests |

No CUDA/MLX pilot. No AutoML. `cut` is not a run kind.
