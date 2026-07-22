import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { AptusDesktopBridge } from "./desktopBridge";

const { bootstrapMock } = vi.hoisted(() => ({
  bootstrapMock: vi.fn(),
}));

vi.mock("./api", async () => {
  const actual = await vi.importActual<typeof import("./api")>("./api");
  return {
    ...actual,
    api: {
      ...actual.api,
      bootstrap: bootstrapMock,
    },
  };
});

import App from "./App";

function installDesktopBridge(): AptusDesktopBridge {
  const bridge: AptusDesktopBridge = {
    platform: "macos",
    reportWorkbenchReady: vi.fn(async () => undefined),
    pickDataset: vi.fn(async () => null),
    pickOutputDirectory: vi.fn(async () => null),
    revealInFinder: vi.fn(async () => undefined),
  };
  window.aptusDesktop = bridge;
  return bridge;
}

beforeEach(() => {
  bootstrapMock.mockReset();
});

afterEach(() => {
  delete window.aptusDesktop;
});

describe("desktop workbench readiness", () => {
  it("reports ready only after authenticated bootstrap commits the stable marker", async () => {
    let resolveBootstrap: ((value: { service: { version: string } }) => void) | undefined;
    bootstrapMock.mockReturnValue(new Promise((resolve) => {
      resolveBootstrap = resolve;
    }));
    const bridge = installDesktopBridge();
    vi.mocked(bridge.reportWorkbenchReady).mockImplementation(async () => {
      expect(document.querySelector("[data-aptus-workbench-ready='aptus-workbench-v1']"))
        .toBeInTheDocument();
    });

    render(<App />);

    expect(bridge.reportWorkbenchReady).not.toHaveBeenCalled();
    expect(document.querySelector("[data-aptus-workbench-ready]")).toBeNull();

    resolveBootstrap?.({ service: { version: "0.2.0" } });

    await waitFor(() => expect(bridge.reportWorkbenchReady).toHaveBeenCalledOnce());
    expect(document.querySelector("[data-aptus-workbench-ready='aptus-workbench-v1']"))
      .toBeInTheDocument();
  });

  it("does not report ready when authenticated bootstrap fails", async () => {
    bootstrapMock.mockRejectedValue(new Error("session rejected"));
    const bridge = installDesktopBridge();

    render(<App />);

    await screen.findByText("The local planner API is unavailable.");
    expect(bridge.reportWorkbenchReady).not.toHaveBeenCalled();
    expect(document.querySelector("[data-aptus-workbench-ready]")).toBeNull();
  });
});
