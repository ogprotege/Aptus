import type { QuantizationLayout } from "../types";

export function formatInspectedQuantizationLayout(
  layout: QuantizationLayout,
): string {
  const defaults = `${layout.default_bits}-bit group ${layout.default_group_size}`;
  const overrides = layout.module_overrides;
  if (overrides.length === 0) {
    return `${defaults}; no overrides`;
  }
  const signatures = overrides.map((item) => {
    const path = item.module_path.replace(/layers\.\d+/g, "layers.N");
    return `${item.bits}-bit ${path}`;
  });
  const unique = [...new Set(signatures)];
  if (unique.length === 1) {
    const count = overrides.length;
    const noun = count === 1 ? "override" : "overrides";
    return `${defaults}; ${count} ${unique[0]} ${noun}`;
  }
  return `${defaults}; ${overrides.length} mixed overrides (${unique.join("; ")})`;
}
