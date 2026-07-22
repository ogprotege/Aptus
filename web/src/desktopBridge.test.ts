import { afterEach, describe, expect, it, vi } from "vitest";
import type { AptusDesktopBridge } from "./desktopBridge";
import { getDesktopBridge } from "./desktopBridge";

afterEach(() => {
  delete window.aptusDesktop;
});

describe("getDesktopBridge", () => {
  it("returns the complete injected desktop bridge", () => {
    const bridge: AptusDesktopBridge = {
      platform: "macos",
      reportWorkbenchReady: vi.fn(async () => undefined),
      pickDataset: vi.fn(async () => "/tmp/training.jsonl"),
      pickOutputDirectory: vi.fn(async () => "/tmp/aptus-output"),
      revealInFinder: vi.fn(async () => undefined),
    };
    window.aptusDesktop = bridge;

    expect(getDesktopBridge()).toBe(bridge);
  });

  it("returns null outside the desktop host", () => {
    expect(getDesktopBridge()).toBeNull();
  });

  it("rejects an incomplete injected object", () => {
    window.aptusDesktop = {
      platform: "macos",
      pickDataset: vi.fn(async () => null),
    } as unknown as AptusDesktopBridge;

    expect(getDesktopBridge()).toBeNull();
  });
});
