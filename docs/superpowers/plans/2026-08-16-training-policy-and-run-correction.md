# Training Policy and Run-Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task **inside one authorized phase only**. Steps use checkbox (`- [ ]`) syntax for tracking. Do not start Phase \(N+1\) until Phase \(N\) exit criteria have evidence.

**Goal:** Give the solo operator a fail-closed, visible instruction-SFT training policy (dataset size, epoch cap, existing rank/alpha/LR/completions-mask priors) and a post-run correction that reads the loss curve as a regularization alarm — never as model quality — without AutoML, Unsloth imports, or a silent rewrite of the operator’s numbers.

**Architecture:** Reuse the existing planner status machine (`feasible` / `conditional` / `infeasible` / `unsupported`), the presentation-only `aptus.plan-correction.v1` object, Compare/CLI surfaces, and M8’s “eval is not loss” split. Add a versioned training-policy prior (`aptus-training-policy-v1`) that can mark candidates conditional or infeasible. Surface existing knob rationale as a second presentation-only object. After a measured run, attach a **new** `aptus.run-correction.v1` object to run metrics — do not add a third kind to plan-correction.v1.

**Tech Stack:** Python planner/refusal/correction (`src/aptus/`), unittest, FastAPI + generated OpenAPI, React Compare stage, CLI stderr presentation, maintained docs under `docs/`.

**Program name:** Training Policy increment (**TP**). This is **not M10**. Mission STATUS after M9 says the M0–M9 program stops; this increment is a new, subordinate workstream that still obeys mission invariants I1–I12.

## Global Constraints

- Invariants I1–I12 in `docs/product/mission-integrity-plan.md` apply to every task.
- Priors are labeled priors. Never say optimal, best, or quality.
- Do not silently rewrite `target.max_epochs` or invent dataset rows.
- Do not change `weight_decay` from `0.0` or add warmup in this increment.
- Completions-only prompt masking already exists; only name it. Do not change the mask contract.
- M8 exact-match eval stays a separate contract. Loss correction must not produce an eval pass/fail.
- Path Alpha (`examples/support-sft.jsonl`, **4 rows**) must remain a **runnable** proof identity. Too-few-rows is **conditional** when `max_epochs <= 3`, not a blanket infeasible.
- Do not bump `aptus-memory-v2` / `aptus-memory-mlx-v2`. Memory arithmetic is unchanged.
- `aptus.plan-correction.v1` kinds stay `select-candidate | no-path`. Run signals get a new schema.
- No Optuna, grid search, Unsloth Desktop, Data Recipes, rsLoRA, LoftQ, BiLoRA/AdaLoRA/DoRA as methods, synthetic data, or GGUF export.
- Presentation objects (`training_policy`, `run_correction`) must not enter `plan_id` / `candidate_id` material, except the explicit `training_policy_version` string added in TP2.
- Owner must approve TP0 before any planner-status code. TP1 (surface existing knobs) may proceed after TP0 because it does not change candidate status.
- Follow `PHASE-PROTOCOL.md`: one phase, completion note, STATUS line, no next-phase files except drafts marked `DRAFT-NOT-AUTHORIZED`.
- Tests: unittest, not pytest. Run `PYTHONPATH=src:. python -m unittest …`.
- Never hand-edit `docs/reference/openapi.v1.json` or `web/src/generated/openapi.ts`; regenerate after API shape changes.
- Claim language in `docs/product/claim-language.md` is updated in the same change as behavior.

---

## Why this increment exists

Aptus already chooses rank, alpha, learning rate, target modules, exact batch arithmetic, and completions-only loss masking. Compare shows the numbers. It does not argue them. `max_epochs` is an unexamined operator fact. `examples/support-sft.jsonl` has four rows. Nothing stops `max_epochs=10` on that file.

M5 already answers “no VRAM / no path.” It does not answer “this loss curve means the next enumerated policy is X.” InstructGPT showed validation loss can “overfit” after one epoch while human preference keeps rising; Biderman 2024 showed LoRA regularizes more than weight decay. So this increment must **watch loss without promoting it to quality**.

Bibliography (do not implement these methods): BiLoRA `2403.13037`, Flexora `2408.10774`, InstructGPT `2203.02155`, BitFit `2106.10199`, LoRA `2106.09685`, Biderman TMLR 2024 `2405.09673` (not in the old PDF pile), Sharma sycophancy `2310.13548`, LIMA `2305.11206`.

---

## File map

| File | Responsibility |
| --- | --- |
| `.superpowers/mission-integrity-plan/TP0-training-policy-spec.md` | Frozen thresholds, identity rules, claim sentences, non-goals. Owner-signed before TP2. |
| `src/aptus/training_policy.py` | Pure functions: classify dataset/epoch, build presentation `training_policy`, later classify run signals. No I/O. |
| `src/aptus/planning.py` | Call policy classifier; append conditional/infeasible reasons; add policy assumptions. No new math for rank/LR/alpha. |
| `src/aptus/refusal.py` | New substring rules → stable reason codes and changeable facts. |
| `src/aptus/correction.py` | Optionally include new codes in plan-correction fact hints. Do not add new kinds. |
| `src/aptus/plan_contract.py` | Put `training_policy_version` into plan identity. Do not hash presentation prose. |
| `src/aptus/api.py` / `api_contracts.py` | Attach `training_policy` on plan responses; later `run_correction` on job/metrics. |
| `src/aptus/cli.py` | Stderr block for training-policy rationale and later run-correction. |
| `src/aptus/execution.py` / job GET | Attach `run_correction` after metrics exist (TP5). |
| `web/src/types.ts`, `web/src/api.ts` | Hand-maintained types + normalizers. |
| `web/src/stages/CompareStage.tsx` | “Why these training knobs” panel. |
| `web/src/stages/RunStage.tsx` or job panel | Run-correction panel (TP5). |
| `tests/aptus/test_training_policy.py` | Classifier unit tests (new). |
| `tests/aptus/test_planning.py` | Status/reason integration. |
| `tests/aptus/test_correction.py` | Plan-correction still identity-safe. |
| `tests/aptus/test_plan_contract.py` | Identity includes policy version; presentation excluded. |
| `web/src/stages/CompareStage.test.tsx` | Panel copy and provenance. |
| `docs/methodology/candidate-enumeration.md` | Rank/alpha/LR already listed; add epoch/dataset rules. |
| `docs/reference/configuration-defaults.md` | Epoch cap and min-rows priors. |
| `docs/guides/choose-a-method.md` | Point 7 already lists knobs; add “why” + epoch/dataset. |
| `docs/product/claim-language.md` | Allowed/forbidden sentences for policy and run-correction. |
| `docs/research/training-policy-bibliography.md` | Extracted bibliography. No PDF binary dump. |
| `docs/product/mission-integrity-plan.md` | Pointer that TP is a post-M9 increment, not M10. |

Do **not** modify: bundle `train.py` mask logic, `weight_decay` / `warmup_steps` compiler defaults, M8 eval schemas, memory formula versions, method catalog, Path Alpha/Beta freeze identities.

---

## Frozen policy (proposed — owner signs in TP0)

These numbers are **instruction-SFT priors**, not measured optima. They apply when `target.task == "sft"` (the only current task).

| Constant | Value | Effect |
| --- | ---: | --- |
| `INSTRUCTION_SFT_MIN_ROWS` | `100` | Below this, not enough supervision for a justified SFT |
| `INSTRUCTION_SFT_EPOCH_CAP` | `3` | Above this, overfitting-risk prior |
| `INSTRUCTION_SFT_SMALL_CORPUS_MAX` | `299` | Inclusive upper bound of the “small corpus” band (`100–299`) |
| `INSTRUCTION_SFT_PARROT_EPOCHS` | `10` | Small corpus + this many epochs is the parrot/sycophant recipe |
| `TRAINING_POLICY_VERSION` | `aptus-training-policy-v1` | Bound into `plan_id` |

### Status rules (do not rewrite operator facts)

| Condition | Candidate status | Reason token (exact planner string) |
| --- | --- | --- |
| `example_count < 100` and `max_epochs <= 3` | **conditional** | `Dataset example_count is below the instruction-SFT supervision prior of 100 rows; this is not a justified domain adaptation.` |
| `example_count < 100` and `max_epochs > 3` | **infeasible** | `Dataset example_count is below 100 rows; Aptus will not endorse training longer than 3 epochs on that set.` |
| `example_count >= 100` and `max_epochs > 3` | **conditional** | `Requested max_epochs exceeds the instruction-SFT epoch-cap prior of 3; Aptus will not rewrite the requested epoch count.` |
| `100 <= example_count < 300` and `max_epochs >= 10` | **infeasible** | `Small instruction corpus (under 300 rows) with max_epochs >= 10 matches the parrot/sycophancy over-training prior.` |
| otherwise | unchanged | no new reason |

Multiple rules may fire. Apply **all** matching reason strings. Status is the **strictest** match (`infeasible` beats `conditional`). Existing memory/runtime reasons still apply independently.

Path Alpha: 4 rows, historically `max_epochs` 1 or 3 → **conditional only**. Still recommended if it was already the viable MLX path. Compare must say the supervision prior out loud.

### What we will not do in TP

- Change rank/alpha/LR formulas.
- Set `weight_decay` to 0.01 or add warmup.
- Early-stop the trainer automatically from val loss (InstructGPT).
- Treat `train_loss < 0.2` as “the model is bad” or “overfit confirmed as quality.”
- Invent rows, mix in ShareGPT, or generate synthetic data.
- Stack a second adapter on an already-adapted identity as a “fix.”
- Search a 12-trial grid.

### Run-signal rules (design in TP4, implement in TP5)

Inputs: existing `train_loss_observations` and `validation_loss_observations` on `metrics.json`. Missing series → `kind: none` (abstain), never a fabricated curve.

| Detection (deterministic) | `kind` | One next enumerated policy |
| --- | --- | --- |
| `len(train) >= 2` and last < first * 0.2 and last < 0.2 | `loss-collapsed` | Decrease `max_epochs` toward the cap, or set alpha = rank (not 2r) on the **next** plan |
| `len(train) >= 2` and last > first * 0.85 | `loss-flat` | Increase `max_epochs` only up to the cap, or review rank prior; never invent decay |
| both series present, first-to-last train decreases, first-to-last eval increases | `eval-rose` | Decrease `max_epochs`; do not emit an M8 eval decision |
| otherwise / insufficient points | `none` | No training-signal correction |

Exact numeric thresholds (`0.2`, `0.85`) are frozen in TP4 after a table of fixture series, not invented in the trainer. They are **heuristics for the next plan**, copied into `non_claims`.

---

## Phase map

```text
TP0  Freeze spec + owner sign-off          [gate]
TP1  Surface existing knobs (no new math)  [presentation]
TP2  Dataset + epoch capability checks     [planner status]
TP3  Compare / CLI / docs for new reasons  [surfaces]
TP4  Run-correction spec freeze            [gate]
TP5  Run-correction objects                [post-run]
TP6  Bibliography + Desktop cleanup        [docs + local files]
```

TP1 may run immediately after TP0. TP2 is the first identity-affecting code. TP3 can share a PR with TP2 if the same branch, but tests must prove each rule before UI copy. TP4 is design-only. TP5 must not start without TP4 approval. TP6 does not touch runtime.

Suggested git: one branch `feat/training-policy` for TP1–TP3; a second branch `feat/run-correction` for TP4–TP5 after TP3 merges.

---

## Phase TP0 — Freeze (no production code)

**Exit:** `TP0-training-policy-spec.md` exists, DECISION log entry exists, owner line signed, STATUS points at TP1.

### Task 0: Write and approve the freeze

**Files:**
- Create: `.superpowers/mission-integrity-plan/TP0-training-policy-spec.md`
- Modify: `.superpowers/mission-integrity-plan/DECISIONS.md`
- Modify: `.superpowers/mission-integrity-plan/STATUS.md`
- Modify: `docs/product/mission-integrity-plan.md` (add a short “post-M9 increments” pointer only)

- [ ] **Step 1:** Copy the “Frozen policy” section of this plan into `TP0-training-policy-spec.md` with Status: Draft until signed.
- [ ] **Step 2:** Append DECISION-20260816-01:

```markdown
## DECISION-20260816-01
- Date: 2026-08-16
- Phase: TP0
- Question: How does Aptus treat <100-row datasets and max_epochs > 3 without breaking Path Alpha or becoming AutoML?
- Options: (a) infeasible below 100 rows always (b) conditional below 100 when epochs<=3; infeasible only for the long-train interactions (c) warn-only, never change status
- Choice: (b) — proposed; owner must confirm.
- Mission justification: Fail-closed on “train longer on too little data”; keep the 4-row Path Alpha proof runnable as conditional, not a silent yes.
- Explicitly will not do: Rewrite max_epochs; invent rows; Optuna; quality badge from loss; weight_decay 0.01; M10 naming.
- Evidence / links: this plan; examples/support-sft.jsonl is 4 rows; BiLoRA; InstructGPT; Biderman 2024.
- Owner: (unsigned)
```

- [ ] **Step 3:** Owner signs the spec and the decision. If the owner rejects (b), stop and rewrite TP2 rules before any planner code.
- [ ] **Step 4:** Set STATUS `current_phase` to TP1. Do not call this M10.

**Human gate:** Do not start TP2 until this signature exists. TP1 is presentation-only and may start after Step 3.

---

## Phase TP1 — Surface existing knobs (no new math)

**Exit:** Compare, CLI, and docs name rank/alpha/LR/completions-mask as **Aptus v0.2 method-class priors**. No candidate `status` change. `plan_id` unchanged for identical facts.

### Task 1: Presentation object and Python tests

**Files:**
- Create: `src/aptus/training_policy.py`
- Create: `tests/aptus/test_training_policy.py`
- Modify: `src/aptus/api.py` (attach field)
- Modify: `src/aptus/api_contracts.py` if a response model is required
- Modify: `src/aptus/cli.py` (`_print_training_policy_block`)

**Interfaces:**
- Consumes: `CandidatePlan` + `TrainingTarget` + `DatasetProfile` already on the plan.
- Produces:

```python
TRAINING_POLICY_SCHEMA_VERSION = "aptus.training-policy.v1"

@dataclass(frozen=True)
class TrainingKnob:
    name: str          # "rank" | "alpha" | "learning_rate" | "completions_mask" | later "epochs" | "dataset_size"
    value: str
    prior_kind: str    # "method-class-prior" | "objective-and-token-volume-prior" | "compiler-contract"
    rationale: str     # one sentence, no quality claim

@dataclass(frozen=True)
class TrainingPolicyPresentation:
    schema_version: str
    policy_version: str   # "aptus-training-policy-v1" even in TP1 (rules not yet applied)
    knobs: tuple[TrainingKnob, ...]
    non_claims: tuple[str, ...]

    def to_primitive(self) -> dict[str, object]: ...

def build_training_policy_presentation(
    *,
    method: str,
    rank: int,
    alpha: int,
    learning_rate: float,
    target_modules: tuple[str, ...],
    example_count: int,
    max_epochs: int,
    truncation_policy: str,
) -> TrainingPolicyPresentation: ...
```

TP1 `build_training_policy_presentation` **does not classify status**. It only explains knobs that already exist.

- [ ] **Step 1: Write the failing tests**

```python
# tests/aptus/test_training_policy.py
class TrainingPolicyPresentationTests(unittest.TestCase):
    def test_adapter_priors_are_labeled_priors_not_optima(self) -> None:
        body = build_training_policy_presentation(
            method="lora",
            rank=16,
            alpha=32,
            learning_rate=2e-4,
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"),
            example_count=4,
            max_epochs=1,
            truncation_policy="completion-first; left-truncate-prompt-to-fit; refuse-empty-supervision",
        )
        text = json.dumps(body.to_primitive()).lower()
        self.assertIn("prior", text)
        self.assertNotIn("optimal", text)
        self.assertNotIn("best", text)
        names = [k.name for k in body.knobs]
        self.assertEqual(
            names,
            ["rank", "alpha", "learning_rate", "completions_mask"],
        )

    def test_full_method_uses_full_lr_prior(self) -> None:
        body = build_training_policy_presentation(
            method="full",
            rank=0,
            alpha=0,
            learning_rate=2e-5,
            target_modules=(),
            example_count=1000,
            max_epochs=1,
            truncation_policy="completion-first; left-truncate-prompt-to-fit; refuse-empty-supervision",
        )
        lr = next(k for k in body.knobs if k.name == "learning_rate")
        self.assertIn("2e-05", lr.value.replace("0.00002", "2e-05"))
        self.assertEqual(lr.prior_kind, "method-class-prior")
```

- [ ] **Step 2:** Run `PYTHONPATH=src:. python -m unittest tests.aptus.test_training_policy -v` — expect FAIL (module missing).
- [ ] **Step 3:** Implement `training_policy.py` with the four knob sentences:

  - rank: “Adapter rank {n} is the Aptus v0.2 objective and dataset-volume prior, not a tuned optimum.”
  - alpha: “Adapter alpha {n} follows the Aptus v0.2 alpha=2*rank policy.”
  - learning_rate: “Learning rate {g} is an Aptus v0.2 method-class prior, not a tuned optimum.”
  - completions_mask: “Loss is computed on assistant/completion tokens only; prompt tokens are masked. Empty supervision is refused.”

  `non_claims` must include: “These knobs are not a prediction of model quality.”

- [ ] **Step 4:** Re-run the unit tests — expect PASS.
- [ ] **Step 5:** Attach `training_policy` on the API plan response the same way `correction` is attached: build after `plan_training`, do not include in `plan_id` material. Add a contract test that two plans with identical facts keep the same `plan_id` when only the presentation object is added.
- [ ] **Step 6:** CLI: print a stderr block titled `Aptus training knobs (presentation only; priors, not optima):` listing the four knobs. Mirror `_emit_correction_block`.
- [ ] **Step 7:** Commit `test: add training-policy presentation tests` then `feat: surface rank alpha lr mask priors`.

### Task 2: Compare UI

**Files:**
- Modify: `web/src/types.ts`
- Modify: `web/src/api.ts` (normalizer; reject unknown schema_version)
- Modify: `web/src/stages/CompareStage.tsx`
- Modify: `web/src/stages/CompareStage.test.tsx` (or colocated test)
- Modify: `web/src/App.css` only if the existing correction-panel classes cannot be reused

- [ ] **Step 1:** Failing test: render a plan with `training_policy` and assert a region named “Why these training knobs” contains “prior” and does not contain “optimal”.
- [ ] **Step 2:** Run `npm --prefix web test -- src/stages/CompareStage.test.tsx` — expect FAIL.
- [ ] **Step 3:** Add a `<section aria-labelledby="training-knobs-title">` **next to** the existing recommendation metrics / inspected gate grid (below CorrectionPanel, above or beside `candidate-contract-grid`). Each knob is a `<div><dt>…</dt><dd>value — rationale</dd></div>`. Provenance badge: inferred/prior, not measured.
- [ ] **Step 4:** Tests pass. `npm --prefix web run typecheck`.
- [ ] **Step 5:** If the HTTP field was added, regenerate OpenAPI from the repo root and `npm --prefix web run openapi:generate`. Run `tools/check_client_contracts.py`.
- [ ] **Step 6:** Commit `feat: show training-knob rationale on Compare`.

### Task 3: Docs for existing knobs

**Files:**
- Modify: `docs/guides/choose-a-method.md` (item 7)
- Modify: `docs/reference/configuration-defaults.md` (compiler-fixed trainer table already lists LR/decay/warmup — add a sentence that Compare names these as priors)
- Modify: `docs/product/claim-language.md` (allowed: “method-class prior”; forbidden: “optimal LoRA rank”)
- Modify: `docs/methodology/candidate-enumeration.md` only if a sentence is needed that the rationale is presentation-only

- [ ] **Step 1:** Update the four docs in the same change. No new behavior.
- [ ] **Step 2:** `PYTHONPATH=src:. python -m unittest tests.aptus.test_documentation -v`
- [ ] **Step 3:** Commit `docs: name training knobs as priors`.

**TP1 completion note:** `.superpowers/mission-integrity-plan/TP1-COMPLETION.md`. Non-claims: we did not add epoch/dataset gates yet.

---

## Phase TP2 — Dataset-size and epoch-cap capability checks

**Exit:** Planner emits the frozen status table. Path Alpha facts (4 rows, `max_epochs<=3`) stay viable-as-conditional. `max_epochs` in the compiled target is unchanged. New `training_policy_version` is in `plan_id`.

### Task 4: Classifier (pure)

**Files:**
- Modify: `src/aptus/training_policy.py`
- Modify: `tests/aptus/test_training_policy.py`

**Interfaces:**
- Produces:

```python
INSTRUCTION_SFT_MIN_ROWS = 100
INSTRUCTION_SFT_EPOCH_CAP = 3
INSTRUCTION_SFT_SMALL_CORPUS_MAX = 299
INSTRUCTION_SFT_PARROT_EPOCHS = 10

@dataclass(frozen=True)
class TrainingPolicyVerdict:
    status: str   # "none" | "conditional" | "infeasible"
    reasons: tuple[str, ...]

def classify_instruction_sft_policy(
    *,
    example_count: int,
    max_epochs: int,
    task: str,
) -> TrainingPolicyVerdict: ...
```

If `task != "sft"`, return `status="none"` and no reasons (forward-safe).

- [ ] **Step 1: Write the failing table tests**

```python
class InstructionSftPolicyTests(unittest.TestCase):
    def test_path_alpha_four_rows_one_epoch_is_conditional_not_infeasible(self) -> None:
        verdict = classify_instruction_sft_policy(
            example_count=4, max_epochs=1, task="sft"
        )
        self.assertEqual(verdict.status, "conditional")
        self.assertTrue(any("below the instruction-SFT supervision prior" in r for r in verdict.reasons))

    def test_four_rows_ten_epochs_is_infeasible(self) -> None:
        verdict = classify_instruction_sft_policy(
            example_count=4, max_epochs=10, task="sft"
        )
        self.assertEqual(verdict.status, "infeasible")

    def test_thousand_rows_five_epochs_is_conditional_and_keeps_requested_count(self) -> None:
        verdict = classify_instruction_sft_policy(
            example_count=1000, max_epochs=5, task="sft"
        )
        self.assertEqual(verdict.status, "conditional")
        self.assertTrue(any("will not rewrite" in r for r in verdict.reasons))

    def test_two_hundred_rows_ten_epochs_is_infeasible_parrot_prior(self) -> None:
        verdict = classify_instruction_sft_policy(
            example_count=200, max_epochs=10, task="sft"
        )
        self.assertEqual(verdict.status, "infeasible")
        self.assertTrue(any("parrot/sycophancy" in r for r in verdict.reasons))

    def test_two_hundred_rows_three_epochs_is_none(self) -> None:
        verdict = classify_instruction_sft_policy(
            example_count=200, max_epochs=3, task="sft"
        )
        self.assertEqual(verdict.status, "none")
        self.assertEqual(verdict.reasons, ())
```

- [ ] **Step 2:** Run the class — expect FAIL.
- [ ] **Step 3:** Implement `classify_instruction_sft_policy` exactly from the frozen table. Do not clamp `max_epochs`.
- [ ] **Step 4:** Tests PASS. Commit `test: classify instruction-SFT dataset and epoch priors`.

### Task 5: Wire into the planner

**Files:**
- Modify: `src/aptus/planning.py` (inside the candidate builder, after existing reason lists, before status assignment — around the current `if unsupported / elif infeasible / elif conditional` block near line 930)
- Modify: `tests/aptus/test_planning.py`

Call site sketch (do not rewrite operator target):

```python
verdict = classify_instruction_sft_policy(
    example_count=dataset.example_count,
    max_epochs=target.max_epochs,
    task=target.task,
)
if verdict.status == "infeasible":
    infeasible.extend(verdict.reasons)
elif verdict.status == "conditional":
    conditional.extend(verdict.reasons)
if verdict.reasons:
    policy_assumptions.append(
        "Instruction-SFT dataset-size and epoch-cap rules are Aptus training-policy v1 priors, not measured quality."
    )
```

- [ ] **Step 1:** Add planning tests that build a minimal plan (reuse existing fixtures) with:
  1. 4 rows, 1 epoch → recommended status is at least conditional; reason present; `target.max_epochs == 1` still.
  2. 4 rows, 10 epochs → no viable candidate (or recommended absent / no-path), reasons present, `target.max_epochs == 10` still on the request/target.
  3. 1000 rows, 5 epochs → viable conditional, epoch not rewritten.
- [ ] **Step 2:** Run `PYTHONPATH=src:. python -m unittest tests.aptus.test_planning tests.aptus.test_training_policy -v` — FAIL then implement then PASS.
- [ ] **Step 3:** Confirm the compiler still writes `max_epochs` from `target.max_epochs` (`generation.py` already copies it). Add a generation test only if one does not already assert pass-through.
- [ ] **Step 4:** Commit `feat: apply instruction-SFT row and epoch priors`.

### Task 6: Refusal catalog + identity version

**Files:**
- Modify: `src/aptus/refusal.py` (append rules; keep substring match style)
- Modify: tests covering `guide_rejection_reason` (existing test file for refusal)
- Modify: `src/aptus/plan_contract.py` plan identity dict
- Modify: `tests/aptus/test_plan_contract.py`
- Modify: `src/aptus/domain.py` or plan constructor if `training_policy_version` is stored on `TrainingPlan`

Identity addition (plan-level, next to `formula_version`):

```python
TRAINING_POLICY_VERSION = "aptus-training-policy-v1"
# in plan identity:
"training_policy_version": plan.get("training_policy_version"),
```

Do **not** hash `training_policy` presentation or `correction`.

Refusal substrings (must match the planner reason strings):

| substring (lowercased) | code | changeable_facts |
| --- | --- | --- |
| `below the instruction-sft supervision prior` | `dataset_below_sft_prior` | `dataset.example_count` |
| `will not endorse training longer than 3 epochs` | `dataset_too_small_for_requested_epochs` | `dataset.example_count`, `target.max_epochs` |
| `exceeds the instruction-sft epoch-cap prior` | `epoch_cap_prior` | `target.max_epochs` |
| `parrot/sycophancy over-training prior` | `small_corpus_high_epoch` | `dataset.example_count`, `target.max_epochs` |

Fact-hint directions: `example_count` → `increase`; `max_epochs` → `decrease` on the infeasible/high-epoch codes, `review` on the supervision-only conditional.

- [ ] **Step 1:** Failing tests for each new code and for “plan_id changes when training_policy_version changes.”
- [ ] **Step 2:** Implement. A plan built **before** this field must fail current-schema validation (same pattern as `formula_version`).
- [ ] **Step 3:** Update `docs/reference/plan-schema.md` with the new plan field.
- [ ] **Step 4:** Commit `feat: bind training-policy version and refusal codes`.

**TP2 completion note:** Record that Path Alpha remains compile-and-run eligible as **conditional**. Explicit non-claim: four rows is not a justified SFT.

---

## Phase TP3 — Compare, CLI, methodology for the new reasons

**Exit:** Operator can see why a candidate is conditional/infeasible for rows/epochs, and the correction CTA still offers one next action (`change-facts` or `confirm-pilot-then-train`).

### Task 7: Surfaces

**Files:**
- Modify: `src/aptus/correction.py` only if new codes should appear in `fact_hints` via existing refusal mapping (preferred: no kind change; hints appear automatically once `refusal.py` maps them)
- Modify: `src/aptus/cli.py` training-policy block to include epoch/dataset knobs now that classification exists
- Modify: `web/src/stages/CompareStage.tsx` — extend the knobs panel with `epochs` and `dataset_size` when the presentation object includes them (add those knobs in `build_training_policy_presentation` now)
- Modify: corresponding tests
- Modify: `docs/methodology/candidate-enumeration.md`
- Modify: `docs/reference/configuration-defaults.md`
- Modify: `docs/guides/choose-a-method.md`
- Modify: `docs/guides/compare-plans.md` if it lists status reasons
- Modify: `docs/product/claim-language.md`

Allowed claim sentences to add:

- “below the instruction-SFT supervision prior of 100 rows”
- “exceeds the instruction-SFT epoch-cap prior of 3”
- “Aptus will not rewrite the requested epoch count”
- “parrot/sycophancy over-training prior”

Forbidden:

- “this dataset will produce a sycophant”
- “3 epochs is optimal”
- “loss proves the model is bad”

- [ ] **Step 1:** Extend `build_training_policy_presentation` with `epochs` and `dataset_size` knobs that quote the verdict when status ≠ none, otherwise “within the instruction-SFT prior.”
- [ ] **Step 2:** UI/CLI tests for the 4-row / 1-epoch fixture: visible “conditional” + supervision prior sentence.
- [ ] **Step 3:** Update methodology docs in the same PR as the surface tests.
- [ ] **Step 4:** Run Python unittest for planning/correction/documentation + `npm --prefix web test` + typecheck + OpenAPI check if the presentation schema grew fields.
- [ ] **Step 5:** Commit `feat: explain dataset and epoch priors on Compare and CLI`.

**TP3 completion note.** Suggest opening PR `feat/training-policy` covering TP1–TP3 if not already stacked.

---

## Phase TP4 — Run-correction spec (no trainer changes)

**Exit:** `.superpowers/mission-integrity-plan/TP4-run-correction-spec.md` approved. No `train.py` edits.

### Task 8: Specify `aptus.run-correction.v1`

This is an M5 **expansion of product behavior**, not a new kind on `aptus.plan-correction.v1`.

```json
{
  "schema_version": "aptus.run-correction.v1",
  "kind": "loss-collapsed | loss-flat | eval-rose | none",
  "summary": "string",
  "source": "train_loss_observations+validation_loss_observations",
  "next_plan_hints": [
    {
      "fact": "target.max_epochs",
      "direction": "decrease",
      "why": "Train loss fell while validation loss rose; next plan should use fewer epochs. This is not an evaluation pass or fail."
    }
  ],
  "disallowed_suggestions": [
    {
      "code": "no_automl",
      "message": "Do not start a hyperparameter search."
    },
    {
      "code": "no_quality_from_loss",
      "message": "Do not treat this signal as model quality or an M8 eval decision."
    },
    {
      "code": "no_weight_decay_as_sycophancy_fix",
      "message": "Do not add weight decay as a sycophancy cure."
    }
  ],
  "operator_next_step": {
    "action": "replan-with-fact-hints | none",
    "label": "string"
  },
  "non_claims": [
    "Training loss is not model quality.",
    "Validation split loss is not an aptus.evaluation-result.v1 decision."
  ]
}
```

Attachment: job/run payload and `metrics.json` companion or job GET field `run_correction`. **Not** in `plan_id`. **Not** a validation level. **Not** required for `measured-run-pass`.

Detection functions live in `training_policy.py` and read only already-recorded finite observation lists. If a series is missing, `kind=none`.

- [ ] **Step 1:** Write TP4 spec with fixture tables (synthetic loss lists) for each kind, including abstain cases (single point, empty, non-finite already rejected by execution).
- [ ] **Step 2:** Freeze numeric cutovers (`0.2`, `0.85`) or replace them with rank-order rules (last vs first) documented in the spec. Prefer order rules plus the existing “below 0.2” Unsloth-adjacent heuristic, labeled heuristic.
- [ ] **Step 3:** Owner approve. STATUS → TP5.

Do not implement in this phase.

---

## Phase TP5 — Run-correction objects

**Exit:** After a full run with metrics, API/CLI/UI can show one training-signal correction. `measured-run-pass` unchanged. M8 eval unchanged.

### Task 9: Classifier + API + CLI

**Files:**
- Modify: `src/aptus/training_policy.py` (`classify_run_loss_signal`)
- Create tests in `tests/aptus/test_training_policy.py`
- Modify: job/metrics response builders (`src/aptus/api.py`, `execution.py` or the job serializer)
- Modify: `src/aptus/cli.py` (job status / inspect-results)
- Modify: OpenAPI generate + `web/src/types.ts` + `web/src/api.ts`
- Modify: Run stage or job panel component + test
- Modify: `docs/guides/inspect-results.md`
- Modify: `docs/product/claim-language.md`

- [ ] **Step 1:** Failing tests for the four kinds using the TP4 fixture lists.
- [ ] **Step 2:** Implement classifier only.
- [ ] **Step 3:** Attach `run_correction` when serving a completed job that has `metrics.json`. Never block `measured-run-pass` on this object.
- [ ] **Step 4:** CLI stderr / inspect-results section. UI panel titled “Training-signal correction (not quality).”
- [ ] **Step 5:** Contract tests: plan_id of the parent plan unchanged; eval-result schema untouched; `non_claims` present.
- [ ] **Step 6:** Regenerated OpenAPI + client checks + web tests + documentation tests.
- [ ] **Step 7:** Commit `feat: add run-correction from recorded loss series`.

**TP5 completion note.** Non-claims: we did not auto-replan, auto-stop training, or change weight decay.

---

## Phase TP6 — Bibliography and Desktop cleanup

**Exit:** Aptus has a maintained bibliography. Desktop `FT-Resources` no longer holds the mixed dump. No runtime change.

### Task 10: Extract, then delete local leftovers

**Files:**
- Create: `docs/research/training-policy-bibliography.md`
- Modify: `docs/research/index.md` (link)
- Local only: `/Users/biscuit/Desktop/FT-Resources/` (owner already has a backup)

Keep in the bibliography (title, arXiv/DOI, what Aptus took, what Aptus refused):

- LoRA `2106.09685`
- BitFit `2106.10199`
- InstructGPT `2203.02155`
- Self-Instruct `2212.10560`
- AdaLoRA `2303.10512` (not implemented)
- LLM-Adapters `2304.01933`
- PEFT survey `2312.12148`
- DoRA `2402.09353` (not implemented)
- BiLoRA `2403.13037` (signal only)
- AFLoRA `2403.13269` (not implemented)
- PEFT survey `2403.14608`
- ReFT `2404.03592` (already research-only)
- ShareLoRA `2406.10785` (not implemented)
- Flexora `2408.10774` (signal only)
- Ultimate Guide `2408.13296` (bibliography; reject its Optuna advice)
- 2025 NLP Journal review `S2949719125000202`
- Biderman 2024 `2405.09673` (missing from the pile; **add by citation**)
- AdamW `1711.05101` (missing; add by citation)
- QLoRA `2305.14314` (missing; add by citation)
- LIMA `2305.11206` (missing; add by citation)
- Sharma sycophancy `2310.13548` (missing; add by citation)

Do **not** copy PDF binaries into the git repo.

- [ ] **Step 1:** Write the bibliography. `test_documentation.py` must pass (links, metadata header).
- [ ] **Step 2:** After the bibliography exists, delete leftover files under `/Users/biscuit/Desktop/FT-Resources/` that are not needed for a local reread. Owner backup already exists. Do not delete anything under the Aptus git worktree except unused untracked copies if present.
- [ ] **Step 3:** Commit `docs: add training-policy bibliography`. Desktop deletion is local, not a git commit.

---

## Spec coverage (self-review)

| Requested item | Phase |
| --- | --- |
| Pre-run dataset-size + epoch-cap in existing status machine | TP2 |
| Visible rejection reasons | TP2 + TP3 |
| Compare “why these training knobs” | TP1 then TP3 |
| Priors named as priors | TP1–TP3 |
| Post-run one primary correction from loss | TP4–TP5 |
| No 12-trial grid | Global + TP4 disallowed |
| Surface rank/alpha/LR/mask, no new math | TP1 |
| Too-few-rows + epoch cap | TP2 |
| Epoch cap 1–3, do not rewrite | TP0/TP2 |
| Dataset × epoch interaction including parrot recipe | TP2 |
| Train/eval divergence as correction only | TP4–TP5 |
| Keep weight_decay 0.0 | Global |
| Keep completions-only masking | TP1 (name only) |
| Separate M8 eval from loss | Global + TP5 tests |
| Phased, don’t destroy Path Alpha / M5 / M8 | TP0 Path Alpha rule; no M10; new run schema |
| FT-Resources extract + delete leftover | TP6 |

## What this plan will not do later either

- Unsloth port
- AutoML
- Quality badge from loss
- Silent epoch rewrite
- Adding 0.01 decay “to stop sycophancy”
- Implementing BiLoRA/Flexora/DoRA/AdaLoRA
- Treating Path Alpha’s 4-row proof as a real SFT

---

## Execution handoff

Plan saved to `docs/superpowers/plans/2026-08-16-training-policy-and-run-correction.md`.

**Do not implement TP2+ until the owner signs TP0**, especially the Path Alpha / `<100 rows` choice (proposed: conditional when `max_epochs <= 3`).

Two execution options after TP0 is signed:

1. **Subagent-driven (recommended)** — one fresh subagent per task, review between tasks, one phase at a time.
2. **Inline** — same sequence in this session with checkpoints at each phase completion note.

**Which approach, and do you approve DECISION-20260816-01 option (b)?**
