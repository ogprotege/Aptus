import { describe, expect, it } from "vitest";
import styles from "./styles.css?raw";

describe("responsive workbench styles", () => {
  it("switches candidate comparison from the content width", () => {
    expect(styles).toContain("container-name: candidate-comparison");
    expect(styles).toContain("@container candidate-comparison (max-width: 1120px)");
    expect(styles).toMatch(/@container candidate-comparison[\s\S]*\.candidate-table-wrap\s*{\s*display: none;/);
    expect(styles).toMatch(/@container candidate-comparison[\s\S]*\.candidate-cards\s*{\s*display: grid;/);
  });

  it("keeps actions in flow and preserves reduced-motion support", () => {
    const actions = styles.match(/\.sticky-actions\s*{([^}]*)}/)?.[1] ?? "";
    expect(actions).not.toContain("position: sticky");
    expect(styles).toContain("@media (prefers-reduced-motion: reduce)");
  });

  it("drives micro-interactions from shared motion tokens", () => {
    expect(styles).toContain("--motion-fast:");
    expect(styles).toContain("--motion-ease:");
    expect(styles).toMatch(/transition:[\s\S]*?var\(--motion-fast\) var\(--motion-ease\)/);
  });

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

  it("locks compound evidence selectors so host borders cannot wipe the stripe", () => {
    expect(styles).toContain(".bundle-contract.evidence-caution");
    expect(styles).toContain(".candidate-card.evidence-path");
    expect(styles).toContain(".candidate-card.evidence-caution");
    expect(styles).toContain(".candidate-card.evidence-blocked");
    expect(styles).toContain(".attestation-panel.evidence-path");
    expect(styles).toContain(".attestation-panel.evidence-omitted");
    expect(styles).toContain(".correction-panel.last-call-door");
    expect(styles).toContain(".correction-panel.last-call-door.evidence-omitted");
    expect(styles).toContain(".candidate-card.evidence-blocked.is-inspected");
  });
});
