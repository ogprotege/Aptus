import { describe, expect, it } from "vitest";
import { EXAMPLE_DRAFT } from "../demo";
import { restoredAvailabilityGiB, summarizeHardwareProbe } from "./hardware";

describe("summarizeHardwareProbe", () => {
  it("keeps unmeasured Apple unified-memory availability unknown", () => {
    const summary = summarizeHardwareProbe(
      {
        devices: [
          {
            name: "Apple M5 Pro (shared unified memory)",
            backend: "mps",
            total_vram_bytes: 64 * 1024 ** 3,
            free_vram_bytes: null,
            supports_bf16: false,
            supports_8bit: false,
            supports_4bit: false,
          },
        ],
        host_ram_bytes: 64 * 1024 ** 3,
        host_ram_free_bytes: null,
        reserve_per_device_bytes: 2 * 1024 ** 3,
        disk_free_bytes: 500 * 1024 ** 3,
      },
      structuredClone(EXAMPLE_DRAFT.hardware),
    );

    expect(summary.devices[0].backend).toBe("mps");
    expect(summary.devices[0].total_vram_gib).toBe(64);
    expect(summary.devices[0].free_vram_gib).toBeNull();
    expect(summary.host_ram_free_gib).toBeNull();
    expect(summary.reserve_per_device_gib).toBe(8);
  });

  it("uses the limiting measured CUDA profile without hiding missing free memory", () => {
    const summary = summarizeHardwareProbe(
      {
        devices: [
          {
            name: "GPU 0",
            backend: "cuda",
            total_vram_gib: 48,
            free_vram_gib: 31,
            supports_bf16: true,
            supports_8bit: true,
            supports_4bit: true,
          },
          {
            name: "GPU 1",
            backend: "cuda",
            total_vram_gib: 48,
            free_vram_gib: null,
            supports_bf16: true,
            supports_8bit: true,
            supports_4bit: false,
          },
        ],
        host_ram_gib: 128,
        reserve_gib: 0,
      },
      structuredClone(EXAMPLE_DRAFT.hardware),
    );

    expect(summary.gpu_count).toBe(2);
    expect(summary.devices[0].free_vram_gib).toBeNull();
    expect(summary.devices[0].supports_4bit).toBe(false);
    expect(summary.reserve_per_device_gib).toBe(0);
  });

  it("does not invent free memory from a partial multi-device restore", () => {
    expect(restoredAvailabilityGiB([null, 20])).toBeNull();
    expect(restoredAvailabilityGiB([18, 20])).toBe(18);
    expect(restoredAvailabilityGiB([])).toBeNull();
  });
});
