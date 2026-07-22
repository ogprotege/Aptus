import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { AptusDesktopBridge } from "./desktopBridge";

const { bootstrapMock, hardwareMock } = vi.hoisted(() => ({
  bootstrapMock: vi.fn(),
  hardwareMock: vi.fn(),
}));

vi.mock("./api", async () => {
  const actual = await vi.importActual<typeof import("./api")>("./api");
  return {
    ...actual,
    api: {
      ...actual.api,
      bootstrap: bootstrapMock,
      hardware: hardwareMock,
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
  hardwareMock.mockReset();
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

  it("describes Apple execution as uninterrupted with no resume after a local scan", async () => {
    bootstrapMock.mockResolvedValue({ service: { version: "0.2.0" } });
    hardwareMock.mockResolvedValue({
      status: "ok",
      scope: "local-measured",
      devices: [{
        name: "Apple M5 Pro (shared unified memory)",
        backend: "mps",
        total_vram_bytes: 64 * 1024 ** 3,
        free_vram_bytes: null,
        supports_bf16: false,
        supports_8bit: false,
        supports_4bit: false,
      }],
      host_ram_bytes: 64 * 1024 ** 3,
      host_ram_free_bytes: 40 * 1024 ** 3,
      reserve_per_device_bytes: 8 * 1024 ** 3,
      disk_free_bytes: 200 * 1024 ** 3,
    });

    render(<App />);

    const scanButton = await screen.findByRole("button", { name: "Scan this Aptus host" });
    fireEvent.click(scanButton);

    const notice = await screen.findByText(
      "Apple Silicon was measured as one shared memory system. Aptus will compare MLX-LM LoRA and QLoRA candidates conservatively. Measured preflight remains a bounded smoke. A passing uninterrupted pilot authorizes an explicitly confirmed full-duration run from scratch; resume is not supported.",
    );
    expect(notice).toBeInTheDocument();
    expect(screen.queryByText(/pilot and full-run approval remain fail-closed/i)).not.toBeInTheDocument();
  });
});
