export const DESKTOP_WORKBENCH_READY_MARKER = "aptus-workbench-v1";

export interface AptusDesktopBridge {
  platform: "macos";
  reportWorkbenchReady: () => Promise<void>;
  pickDataset: () => Promise<string | null>;
  pickOutputDirectory: () => Promise<string | null>;
  revealInFinder: (path: string) => Promise<void>;
}

declare global {
  interface Window {
    aptusDesktop?: AptusDesktopBridge;
  }
}

function isDesktopBridge(value: unknown): value is AptusDesktopBridge {
  if (!value || typeof value !== "object") return false;

  const bridge = value as Partial<AptusDesktopBridge>;
  return (
    bridge.platform === "macos"
    && typeof bridge.reportWorkbenchReady === "function"
    && typeof bridge.pickDataset === "function"
    && typeof bridge.pickOutputDirectory === "function"
    && typeof bridge.revealInFinder === "function"
  );
}

export function getDesktopBridge(): AptusDesktopBridge | null {
  if (typeof window === "undefined" || !isDesktopBridge(window.aptusDesktop)) {
    return null;
  }
  return window.aptusDesktop;
}
