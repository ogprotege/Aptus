import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { EXAMPLE_BUNDLE, EXAMPLE_PLAN } from "../demo";
import type { AptusDesktopBridge } from "../desktopBridge";
import { CompileStage } from "./CompileStage";

afterEach(() => {
  delete window.aptusDesktop;
});

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
    expect(screen.queryByRole("button", { name: /Finder/i })).not.toBeInTheDocument();
  });

  it("chooses an output folder through the desktop bridge", async () => {
    const onOutputDirChange = vi.fn();
    const bridge: AptusDesktopBridge = {
      platform: "macos",
      reportWorkbenchReady: vi.fn(async () => undefined),
      pickDataset: vi.fn(async () => null),
      pickOutputDirectory: vi.fn(async () => "/Users/wilson/Aptus Bundles/new-plan"),
      revealInFinder: vi.fn(async () => undefined),
    };
    window.aptusDesktop = bridge;
    render(
      <CompileStage
        plan={EXAMPLE_PLAN}
        bundle={null}
        busy={null}
        demoMode={false}
        onCompile={vi.fn(async () => undefined)}
        onValidate={vi.fn(async () => undefined)}
        onReturnToCompare={vi.fn()}
        outputDir="/tmp/aptus-output"
        onOutputDirChange={onOutputDirChange}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Choose folder" }));

    await waitFor(() => {
      expect(onOutputDirChange).toHaveBeenCalledWith("/Users/wilson/Aptus Bundles/new-plan");
    });
  });

  it("reveals the compiled bundle and archive in Finder", async () => {
    const bridge: AptusDesktopBridge = {
      platform: "macos",
      reportWorkbenchReady: vi.fn(async () => undefined),
      pickDataset: vi.fn(async () => null),
      pickOutputDirectory: vi.fn(async () => null),
      revealInFinder: vi.fn(async () => undefined),
    };
    window.aptusDesktop = bridge;
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

    fireEvent.click(screen.getByRole("button", { name: "Show bundle in Finder" }));
    if (EXAMPLE_BUNDLE.archive_path) {
      fireEvent.click(screen.getByRole("button", { name: "Show archive in Finder" }));
    }

    await waitFor(() => {
      expect(bridge.revealInFinder).toHaveBeenCalledWith(EXAMPLE_BUNDLE.bundle_dir);
      if (EXAMPLE_BUNDLE.archive_path) {
        expect(bridge.revealInFinder).toHaveBeenCalledWith(EXAMPLE_BUNDLE.archive_path);
      }
    });
  });
});
