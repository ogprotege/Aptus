# Lane 4 Workbench Craft Implementation Plan

> **Status:** Active | **Authority:** Implementation plan (subordinate to claim language and Lane 4 spec freeze) | **Applies to:** Aptus Lane 4 workbench-craft increment, not M10, not 0.3, not more 0.2 | **Audience:** Agents executing Tasks 1–7 | **Last reviewed:** 2026-08-19 | **Review by:** After Tasks 1–7 merge

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the five-stage React workbench look as honest as the frozen 0.2 referee already is, without new verbs, new stages, or Unsloth “you won” chrome.

**Architecture:** Keep Familjen Grotesk / Atkinson Hyperlegible / IBM Plex Mono and teal/amber/red meanings. Add one spacing scale, one type scale, and four evidence-state classes in `web/src/styles.css`. Shared components (`StatusBadge`, `EmptyStage`, `StageHeader`, `RunConsole`) consume those classes. Stages do not change what they *do*. Packaged `src/aptus/_web/` rebuilds with the visual change.

**Tech Stack:** React + Vite workbench, Vitest + Testing Library, existing CSS tokens, colocated `*.test.tsx`. No Python contract changes.

## Global Constraints

- Spec freeze: `docs/superpowers/specs/2026-08-19-lane-4-workbench-craft-design.md`
- Increment name is Lane 4. Not M10. Not 0.3. Not a 0.2 ship. Do not grow 0.2.
- Surfaces: React five-stage workbench only. Not Mac Home / Machine / Models.
- Keep type + teal (`--circuit-teal`) / amber (`--calibration-amber`) / red (`--fault-red`) meanings.
- One pass across Facts, Compare, Compile, Validate, and Run.
- Unsloth-style Train chrome is forbidden: no celebration motion, no “your model is ready.”
- No new product verbs. Copy stays inside claim language and `docs/product/ui-ux.md`.
- Do not change Python planner, compiler, JobService, or HTTP contracts.
- Never hand-edit `docs/reference/openapi.v1.json` or `web/src/generated/openapi.ts`.
- Motion: existing `--motion-fast` (150ms) and `prefers-reduced-motion` only.
- Color is never the only status signal.
- No CUDA/MLX pilot.
- Do not commit `aptus-work/` or `web/node_modules`.
- Web tests: `npm --prefix web test` (Vitest). Not pytest. Not `python -m unittest` except documentation locks.

---

## File map

| File | Responsibility |
| --- | --- |
| `web/src/styles.css` | Spacing scale, type scale, `.evidence-*`, `.last-call-door` |
| `web/src/styles.test.ts` | Token and class presence; reduced-motion still present |
| `web/src/components/StatusBadge.tsx` | Map omitted/absent to `omitted` tone |
| `web/src/components/EmptyStage.tsx` | Optional `tone`: `path` (default) or `omitted` |
| `web/src/components/EmptyStage.test.tsx` | Create: omitted empty is not a primary win |
| `web/src/components/RunConsole.tsx` | Last-call panel uses `.last-call-door`; missing is omitted |
| `web/src/stages/*.tsx` | Apply evidence classes; do not change stage jobs |
| `web/src/stages/CompareStage.test.tsx` | Unsupported stays visible; blocked class |
| `web/src/stages/RunStage.test.tsx` | No last call recorded looks omitted |
| `docs/product/ui-ux.md` | Craft sentences + last reviewed |
| `docs/product/claim-language.md` | Optional lock: “craft is not a quality yes” |
| `src/aptus/_web/` | Regenerated packaged assets |
| `tests/aptus/test_documentation.py` | Inventory counts if docs added |

Do **not** modify: `planning.py`, `plan_contract.py`, `execution.py`, `run_disposition.py`, bundle `train.py`, SwiftUI shell, OpenAPI artifacts.

---

### Task 1: Craft tokens

**Files:**
- Modify: `web/src/styles.css` (inside `:root` after `--motion-ease`, and a new “Lane 4 evidence states” block after `.status-badge`)
- Test: `web/src/styles.test.ts`

**Interfaces:**
- Consumes: existing `--circuit-teal`, `--calibration-amber`, `--fault-red`, `--motion-fast`, `--font-display`, `--font-body`, `--font-mono`
- Produces (exact names later tasks must use):
  - `--space-1: 4px;` `--space-2: 8px;` `--space-3: 12px;` `--space-4: 16px;` `--space-5: 24px;` `--space-6: 32px;`
  - `--type-display: 1.75rem;` `--type-title: 1.25rem;` `--type-body: 1rem;` `--type-meta: 0.875rem;`
  - `.evidence-path` `.evidence-caution` `.evidence-blocked` `.evidence-omitted`
  - `.last-call-door`
  - `.status-omitted` (badge tone)

- [ ] **Step 1: Write the failing test**

In `web/src/styles.test.ts`, add:

```ts
  it("locks Lane 4 craft tokens and evidence-state classes", () => {
    expect(styles).toContain("--space-1: 4px");
    expect(styles).toContain("--space-6: 32px");
    expect(styles).toContain("--type-display:");
    expect(styles).toContain("--type-meta:");
    expect(styles).toContain(".evidence-path");
    expect(styles).toContain(".evidence-caution");
    expect(styles).toContain(".evidence-blocked");
    expect(styles).toContain(".evidence-omitted");
    expect(styles).toContain(".last-call-door");
    expect(styles).toContain(".status-omitted");
    expect(styles).toContain("@media (prefers-reduced-motion: reduce)");
    expect(styles).not.toContain("confetti");
  });
```

- [ ] **Step 2: Run it to see fail**

```bash
cd web && npx vitest run src/styles.test.ts
```

Expected: FAIL — `--space-1` not found.

- [ ] **Step 3: Add tokens and classes**

In `:root` after `--motion-ease`:

```css
  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-5: 24px;
  --space-6: 32px;
  --type-display: 1.75rem;
  --type-title: 1.25rem;
  --type-body: 1rem;
  --type-meta: 0.875rem;
```

After `.status-badge` block, add:

```css
.status-omitted,
.evidence-omitted {
  color: var(--muted);
  border-color: var(--line);
  background: var(--faint);
}

.evidence-path {
  border-left: 3px solid var(--circuit-teal);
}

.evidence-caution {
  border-left: 3px solid var(--calibration-amber);
}

.evidence-blocked {
  border-left: 3px solid var(--fault-red);
}

.last-call-door {
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: var(--space-5);
  background: var(--porcelain);
}

.last-call-door.evidence-omitted {
  border-style: dashed;
}
```

Do not add keyframe celebrations. Keep `@media (prefers-reduced-motion: reduce)`.

Point `.stage-header h1` at `var(--type-display)` and `.stage-lede` / `.eyebrow` at `var(--type-meta)` / existing eyebrow rules without inventing a second type system.

- [ ] **Step 4: Run test to verify it passes**

```bash
cd web && npx vitest run src/styles.test.ts
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/src/styles.css web/src/styles.test.ts
git commit -m "feat: add Lane 4 workbench craft tokens"
```

---

### Task 2: Shared status and empty tones

**Files:**
- Modify: `web/src/components/StatusBadge.tsx`
- Modify: `web/src/components/EmptyStage.tsx`
- Create: `web/src/components/EmptyStage.test.tsx`
- Modify: `web/src/components/StatusBadge` has no dedicated test file — add `web/src/components/StatusBadge.test.tsx`

**Interfaces:**
- Consumes: `.status-omitted`, `.evidence-omitted` from Task 1
- Produces:
  - `StatusBadge` `tone` includes `omitted` when `state` lowercases to `omitted`, `absent`, or `no-last-call`
  - `EmptyStage` props: existing plus optional `tone?: "path" | "omitted"` (default `"path"`)
  - When `tone="omitted"`: `className="empty-stage evidence-omitted"` and the button uses `button-secondary`, not `button-primary`

- [ ] **Step 1: Write the failing tests**

`web/src/components/StatusBadge.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { StatusBadge } from "./StatusBadge";

describe("StatusBadge", () => {
  it("maps omitted states to the omitted tone", () => {
    const { container } = render(<StatusBadge state="omitted" />);
    expect(container.firstChild).toHaveClass("status-omitted");
    expect(screen.getByText("omitted")).toBeInTheDocument();
  });

  it("keeps unsupported on the negative tone", () => {
    const { container } = render(<StatusBadge state="unsupported" />);
    expect(container.firstChild).toHaveClass("status-negative");
  });

  it("keeps conditional on the warning tone", () => {
    const { container } = render(<StatusBadge state="conditional" />);
    expect(container.firstChild).toHaveClass("status-warning");
  });
});
```

`web/src/components/EmptyStage.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { EmptyStage } from "./EmptyStage";

describe("EmptyStage", () => {
  it("uses a secondary action when the empty state is omitted", () => {
    render(
      <EmptyStage title="No last call recorded." actionLabel="Stay on Run" onAction={() => undefined} tone="omitted">
        Missing last call is not Use.
      </EmptyStage>,
    );
    const section = screen.getByText("No last call recorded.").closest("section");
    expect(section).toHaveClass("evidence-omitted");
    expect(screen.getByRole("button", { name: "Stay on Run" })).toHaveClass("button-secondary");
    expect(screen.getByRole("button", { name: "Stay on Run" })).not.toHaveClass("button-primary");
  });

  it("keeps the default empty stage on the path button", () => {
    const onAction = vi.fn();
    render(
      <EmptyStage title="Need a plan" actionLabel="Back to Facts" onAction={onAction}>
        Compile waits on a plan.
      </EmptyStage>,
    );
    expect(screen.getByRole("button", { name: "Back to Facts" })).toHaveClass("button-primary");
  });
});
```

- [ ] **Step 2: Run them to see fail**

```bash
cd web && npx vitest run src/components/StatusBadge.test.tsx src/components/EmptyStage.test.tsx
```

Expected: FAIL — `status-omitted` missing; `tone` is not a prop.

- [ ] **Step 3: Minimal implementation**

In `StatusBadge.tsx`, add `"omitted", "absent", "no-last-call"` to a new branch that sets `tone = "omitted"` (class `status-omitted`). Keep `unsupported`/`infeasible` on `negative` and `conditional` on `warning`.

In `EmptyStage.tsx`:

```tsx
interface EmptyStageProps {
  title: string;
  children: ReactNode;
  actionLabel: string;
  onAction: () => void;
  tone?: "path" | "omitted";
}

export function EmptyStage({ title, children, actionLabel, onAction, tone = "path" }: EmptyStageProps) {
  const omitted = tone === "omitted";
  return (
    <section className={omitted ? "empty-stage evidence-omitted" : "empty-stage"}>
      <span className="empty-glyph" aria-hidden="true">⌁</span>
      <h2>{title}</h2>
      <p>{children}</p>
      <button type="button" className={omitted ? "button button-secondary" : "button button-primary"} onClick={onAction}>
        {actionLabel}
      </button>
    </section>
  );
}
```

Do not remove the action button. Do not add celebration glyphs.

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd web && npx vitest run src/components/StatusBadge.test.tsx src/components/EmptyStage.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/src/components/StatusBadge.tsx web/src/components/StatusBadge.test.tsx \
  web/src/components/EmptyStage.tsx web/src/components/EmptyStage.test.tsx
git commit -m "feat: give omitted empty and status a non-path tone"
```

---

### Task 3: Facts craft

**Files:**
- Modify: `web/src/stages/FactsStage.tsx`
- Test: `web/src/stages/FactsStage.test.tsx` (extend)

**Interfaces:**
- Consumes: `.evidence-caution`, ProvenanceBadge (unchanged kinds)
- Produces: inspection result region uses `evidence-caution` when present; example mode stays labeled. No new facts. Inspection still cannot check training permission.

- [ ] **Step 1: Write a failing test** in `FactsStage.test.tsx`

The file already uses `<FactsHarness />`. After the existing “does not present evaluation fraction as a quality contract” test, add:

```tsx
  it("marks the evaluation-fraction field as caution, not a quality path", () => {
    render(<FactsHarness />);
    const field = screen.getByLabelText("Evaluation fraction").closest(".field");
    expect(field).toHaveClass("evidence-caution");
  });
```

- [ ] **Step 2: Run it to see fail** if a new class is missing:

```bash
cd web && npx vitest run src/stages/FactsStage.test.tsx
```

- [ ] **Step 3: Apply the class**

On the Facts evaluation-fraction `.field` wrapper, add `evidence-caution` next to `field`. Do not change form submit, inspection receipt clearing, hardware scan, or the accessible description copy.

- [ ] **Step 4: Re-run Facts tests**

```bash
cd web && npx vitest run src/stages/FactsStage.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/src/stages/FactsStage.tsx web/src/stages/FactsStage.test.tsx
git commit -m "style: apply Lane 4 craft on Facts"
```

---

### Task 4: Compare craft

**Files:**
- Modify: `web/src/stages/CompareStage.tsx`
- Modify: `web/src/components/CandidateComparison.tsx` (row/card container class from status)
- Modify: `web/src/components/FitLedger.tsx` only if a wrapper is needed; prefer CSS on existing copy “point estimate” / “heuristic upper”
- Test: `web/src/stages/CompareStage.test.tsx`
- Test: `web/src/components/CandidateComparison.test.tsx` if that file already covers status rows

**Interfaces:**
- Consumes: `.evidence-path` `.evidence-caution` `.evidence-blocked` from Task 1
- Produces: candidate row/card `className` includes `evidence-path` when `status === "feasible"`, `evidence-caution` when `conditional`, `evidence-blocked` when `infeasible` or `unsupported`. Unsupported rows remain in the document.

- [ ] **Step 1: Write the failing test**

In `CompareStage.test.tsx`, next to `rejected` (status `infeasible`), add:

```tsx
const unsupported: CandidatePlan = {
  ...rejected,
  candidate_id: `cand_${"d".repeat(20)}`,
  status: "unsupported",
  rejection_reasons: ["The method registry does not list this distribution."],
};
```

Use the same `CompareStage` props as `does not call a missing recommendation a safe plan`, but `plan={{ ...noFeasiblePlan, candidates: [unsupported] }}` and `selected={unsupported}`:

```tsx
  it("keeps unsupported candidates visible with the blocked evidence class", () => {
    render(
      <CompareStage
        plan={{ ...noFeasiblePlan, candidates: [unsupported] }}
        selected={unsupported}
        busy={null}
        demoMode={false}
        modelPolicyPresentation={null}
        onInspectCandidate={vi.fn()}
        onSelectCandidate={vi.fn(async () => undefined)}
        onCompile={vi.fn(async () => undefined)}
        onReturnToFacts={vi.fn()}
      />,
    );
    expect(screen.getByText("unsupported")).toBeInTheDocument();
    expect(document.querySelector(".evidence-blocked")).not.toBeNull();
  });
```

- [ ] **Step 2: Run it to see fail**

```bash
cd web && npx vitest run src/stages/CompareStage.test.tsx
```

Expected: FAIL — `.evidence-blocked` missing.

- [ ] **Step 3: Map status to class**

Helper in `CandidateComparison.tsx` (or CompareStage if cards are there):

```ts
function evidenceClass(status: string): string {
  if (status === "feasible") return "evidence-path";
  if (status === "conditional") return "evidence-caution";
  if (status === "infeasible" || status === "unsupported") return "evidence-blocked";
  return "evidence-omitted";
}
```

Put it on the row/card root. Do not hide unsupported. Do not change ranking, recommendation copy, or policy-panel decoding.

Keep visible labels “Point estimate” and “Heuristic upper envelope” as separate fields.

- [ ] **Step 4: Run tests**

```bash
cd web && npx vitest run src/stages/CompareStage.test.tsx src/components/CandidateComparison.test.tsx src/components/FitLedger.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/src/stages/CompareStage.tsx web/src/stages/CompareStage.test.tsx \
  web/src/components/CandidateComparison.tsx web/src/components/CandidateComparison.test.tsx \
  web/src/components/FitLedger.tsx
git commit -m "style: mark Compare path, caution, and blocked"
```

Stage only files you actually changed.

---

### Task 5: Compile and Validate craft

**Files:**
- Modify: `web/src/stages/CompileStage.tsx`
- Modify: `web/src/stages/ValidateStage.tsx`
- Test: `web/src/stages/CompileStage.test.tsx`
- Test: `web/src/stages/ValidateStage.test.tsx`

**Interfaces:**
- Consumes: Task 1 classes; Task 2 `EmptyStage` `tone`
- Produces: empty compile (no plan) stays an empty stage with default `tone="path"` (action is “back”, not a win). Validate unbound evidence region uses `evidence-omitted`. Success copy remains “bundle written” / evidence rungs — do not add “ready to ship.”

- [ ] **Step 1: Write failing tests**

In `CompileStage.test.tsx`, extend the existing `EXAMPLE_PLAN` + `EXAMPLE_BUNDLE` render:

```tsx
  it("marks the generated-code boundary as caution, not a ship", () => {
    render(
      <CompileStage
        plan={EXAMPLE_PLAN}
        bundle={EXAMPLE_BUNDLE}
        busy={null}
        demoMode={false}
        onCompile={vi.fn(async () => undefined)}
        onValidate={vi.fn(async () => undefined)}
        onReturnToCompare={vi.fn()}
        outputDir="/tmp/aptus-output"
        onOutputDirChange={vi.fn()}
      />,
    );
    expect(screen.queryByText(/ready to ship/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/your model is ready/i)).not.toBeInTheDocument();
    expect(screen.getByText("Generated-code boundary").closest("section")).toHaveClass(
      "evidence-caution",
    );
  });
```

In `ValidateStage.test.tsx`, add to `renders attestation bindings and both pilot phases` (that fixture already shows “Bound validation evidence”):

```tsx
    expect(screen.getByText("Bound validation evidence").closest("section")).toHaveClass(
      "evidence-path",
    );
```

- [ ] **Step 2: Run to see fail**

```bash
cd web && npx vitest run src/stages/CompileStage.test.tsx src/stages/ValidateStage.test.tsx
```

- [ ] **Step 3: Apply classes** on empty compile, generated-code boundary (caution, not path), and unbound validation evidence. Do not change no-clobber, path picking, or validation API calls.

- [ ] **Step 4: Re-run**

```bash
cd web && npx vitest run src/stages/CompileStage.test.tsx src/stages/ValidateStage.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/src/stages/CompileStage.tsx web/src/stages/CompileStage.test.tsx \
  web/src/stages/ValidateStage.tsx web/src/stages/ValidateStage.test.tsx
git commit -m "style: apply Lane 4 craft on Compile and Validate"
```

---

### Task 6: Run last-call door

**Files:**
- Modify: `web/src/components/RunConsole.tsx` (the disposition `<section>` currently `className="correction-panel"`)
- Test: `web/src/stages/RunStage.test.tsx`

**Interfaces:**
- Consumes: `.last-call-door`, `.evidence-omitted` from Task 1
- Produces: the last-call `<section>` has `className="correction-panel last-call-door"` and, when `disposition` is missing, also `evidence-omitted`. Buttons stay `button-secondary`. Copy “No last call recorded.” remains. No `button-primary` on Use.

- [ ] **Step 1: Extend the existing last-call test**

In `RunStage.test.tsx`, inside `shows operator-attested Use/Done/Stop on a completed train without a last call` (job `job_last_call`, no `run_disposition`), add:

```tsx
    expect(region).toHaveTextContent("No last call recorded.");
    expect(region).toHaveClass("last-call-door");
    expect(region).toHaveClass("evidence-omitted");
    expect(screen.getByRole("button", { name: "Use it" })).toHaveClass("button-secondary");
    expect(screen.getByRole("button", { name: "Use it" })).not.toHaveClass("button-primary");
```

`region` is already `getByRole("region", { name: "What do you want to do with what you just trained?" })`. Do not invent a new job shape.

- [ ] **Step 2: Run it to see fail**

```bash
cd web && npx vitest run src/stages/RunStage.test.tsx
```

Expected: FAIL — `last-call-door` missing.

- [ ] **Step 3: Implement**

On the disposition section in `RunConsole.tsx`:

```tsx
    <section
      className={disposition ? "correction-panel last-call-door" : "correction-panel last-call-door evidence-omitted"}
      aria-labelledby="run-disposition-title"
    >
```

Keep the three secondary buttons. Do not add charts or “your model is ready.”

- [ ] **Step 4: Re-run**

```bash
cd web && npx vitest run src/stages/RunStage.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/src/components/RunConsole.tsx web/src/stages/RunStage.test.tsx
git commit -m "style: present Use/Done/Stop as a last-call door"
```

---

### Task 7: Contract docs, a11y, packaged workbench

**Files:**
- Modify: `docs/product/ui-ux.md` (last reviewed → 2026-08-19; add a short “Lane 4 craft” paragraph under Five stages or Accessibility: craft is not a stronger evidence level; last call missing is omitted)
- Modify: `docs/product/claim-language.md` — add allowed: “craft is not a quality yes”
- Modify: `tests/aptus/test_documentation.py` — extend `test_claim_language_locks_lane_3_run_disposition_sentences` to also require `craft is not a quality yes`
- Modify: `web/src/accessibility.test.tsx` only if a new name/role would break axe; keep it green
- Regenerate: `src/aptus/_web/` via `npm --prefix web run build`

**Interfaces:** none (docs + packaged assets)

- [ ] **Step 1: Write the failing documentation test**

In `tests/aptus/test_documentation.py` `test_claim_language_locks_lane_3_run_disposition_sentences`:

```python
        self.assertIn("craft is not a quality yes", claim_language)
```

- [ ] **Step 2: Run it to see fail**

```bash
PYTHONPATH=src:. python -m unittest tests.aptus.test_documentation.DocumentationTests.test_claim_language_locks_lane_3_run_disposition_sentences -v
```

Expected: FAIL — phrase missing.

- [ ] **Step 3: Add the sentence** to `docs/product/claim-language.md` under Run disposition claims allowed list. Update `docs/product/ui-ux.md` last reviewed and one paragraph: Lane 4 tightens craft; empty last call is omitted; color is still not the only status signal.

- [ ] **Step 4: Run docs + web gates**

```bash
PYTHONPATH=src:. python -m unittest tests.aptus.test_documentation -v
cd web && npm test && npm run typecheck && npm run build
```

Expected: documentation module PASS;  web tests PASS; typecheck PASS; build writes `src/aptus/_web/`.

If inventory counts change because you add no new markdown files, do not bump them. This task should not add a new `.md` path besides edits to existing files.

- [ ] **Step 5: Commit** (include packaged assets; never `web/node_modules`)

```bash
git add docs/product/ui-ux.md docs/product/claim-language.md \
  tests/aptus/test_documentation.py src/aptus/_web web/src/accessibility.test.tsx
git commit -m "docs: lock Lane 4 craft language and package workbench"
```

---

## Spec coverage

| Spec § | Task |
| --- | --- |
| 1 Goal | all |
| 2 Non-goals | Global Constraints |
| 3 Surfaces | all (React only) |
| 4 Visual identity | Task 1 |
| 5 Craft rubric | Tasks 1–6 |
| 6 Stage jobs | Tasks 3–6 (no behavior change) |
| 7 Files | file map |
| 8 Testing | each task + Task 7 |
| 9 Claim language | Task 7 |
| 10 Frozen decisions | Global Constraints |
| 11 Success | Tasks 4 and 6 (blocked vs omitted vs door) |

No CUDA/MLX pilot. No native shell. No 0.3 version bump.
