import type { HardwareFacts, HardwareProbeResponse } from "../types";

function finiteNumber(
  value: unknown,
  { allowZero = false }: { allowZero?: boolean } = {},
): number | null {
  if (typeof value !== "number" || !Number.isFinite(value)) return null;
  if (allowZero ? value < 0 : value <= 0) return null;
  return value;
}

function gibibytes(
  directValue: unknown,
  byteValue: unknown,
  { allowZero = false }: { allowZero?: boolean } = {},
): number | null {
  const direct = finiteNumber(directValue, { allowZero });
  if (direct !== null) return direct;
  const bytes = finiteNumber(byteValue, { allowZero });
  return bytes === null ? null : bytes / 1024 ** 3;
}

/**
 * Convert measured local-host facts into the single limiting profile used by
 * the v0.2 workbench. Missing availability stays unknown. In particular, the
 * total Apple unified-memory pool must never be presented as currently free.
 */
export function restoredAvailabilityGiB(values: Array<number | null>): number | null {
  if (!values.length || values.some((value) => value === null)) {
    return null;
  }
  return Math.min(...(values as number[]));
}

export function summarizeHardwareProbe(
  probe: HardwareProbeResponse,
  current: HardwareFacts,
): HardwareFacts {
  const scannedDevices = probe.devices ?? [];
  if (!scannedDevices.length) {
    throw new Error("The hardware probe returned no supported accelerator records.");
  }
  const backends = new Set(
    scannedDevices
      .map((device) => device.backend)
      .filter((backend): backend is string => Boolean(backend)),
  );
  if (
    backends.size !== 1
    || scannedDevices.some((device) => !device.backend)
  ) {
    throw new Error("The hardware probe returned a missing or mixed backend.");
  }
  const backend = [...backends][0] as string;
  const totals = scannedDevices.map((device) =>
    gibibytes(device.total_vram_gib, device.total_vram_bytes),
  );
  if (totals.some((value) => value === null)) {
    throw new Error("The hardware probe did not return usable total memory measurements.");
  }
  const measuredTotals = totals as number[];
  const freeValues = scannedDevices.map((device) =>
    gibibytes(device.free_vram_gib, device.free_vram_bytes),
  );
  const allAvailabilityMeasured = freeValues.every(
    (value): value is number => value !== null,
  );
  const measuredFreeValues = allAvailabilityMeasured
    ? (freeValues as number[])
    : null;
  const limitingIndex = measuredTotals.reduce(
    (currentIndex, value, index) =>
      value < measuredTotals[currentIndex] ? index : currentIndex,
    0,
  );
  const hostRam = gibibytes(probe.host_ram_gib, probe.host_ram_bytes);
  if (hostRam === null) {
    throw new Error("The hardware probe did not return usable host-memory measurements.");
  }

  return {
    ...current,
    discovery: "local-scan",
    gpu_count: scannedDevices.length,
    host_ram_gib: hostRam,
    host_ram_free_gib: gibibytes(
      probe.host_ram_free_gib,
      probe.host_ram_free_bytes,
    ),
    reserve_per_device_gib:
      backend === "mps"
        ? Math.max(
            gibibytes(
              probe.reserve_gib,
              probe.reserve_per_device_bytes,
              { allowZero: true },
            ) ?? current.reserve_per_device_gib ?? 0,
            8,
          )
        : gibibytes(
            probe.reserve_gib,
            probe.reserve_per_device_bytes,
            { allowZero: true },
          ) ?? current.reserve_per_device_gib,
    disk_free_gib: gibibytes(probe.disk_free_gib, probe.disk_free_bytes),
    devices: [
      {
        name:
          scannedDevices.length === 1
            ? scannedDevices[limitingIndex]?.name ?? "Scanned accelerator"
            : `Distributed limiting profile across ${scannedDevices.length} accelerators`,
        backend,
        total_vram_gib: Math.min(...measuredTotals),
        free_vram_gib: measuredFreeValues
          ? Math.min(...measuredFreeValues)
          : null,
        supports_bf16: scannedDevices.every(
          (device) => device.supports_bf16 === true,
        ),
        supports_8bit: scannedDevices.every(
          (device) => device.supports_8bit === true,
        ),
        supports_4bit: scannedDevices.every(
          (device) => device.supports_4bit === true,
        ),
      },
    ],
  };
}
