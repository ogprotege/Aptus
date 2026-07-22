import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { EXAMPLE_BUNDLE, EXAMPLE_PLAN } from "../demo";
import { CompileStage } from "./CompileStage";

describe("CompileStage", () => {
  it("states the exact dependency contract beside a compiled bundle", () => {
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

    const note = screen.getByRole("note");
    expect(note).toHaveTextContent("requirements.txt");
    expect(note).toHaveTextContent("exact direct package pins");
    expect(note).toHaveTextContent("not a complete transitive lock");
  });
});
