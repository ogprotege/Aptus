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
});
