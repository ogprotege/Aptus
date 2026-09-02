import { describe, expect, it } from "vitest";
import { formatInspectedQuantizationLayout } from "./quantizationLayout";

describe("formatInspectedQuantizationLayout", () => {
  it("names Gemma router.proj 8-bit overrides instead of a collapsed count", () => {
    const layout = {
      default_bits: 4,
      default_group_size: 64,
      module_overrides: Array.from({ length: 30 }, (_, index) => ({
        module_path: `model.layers.${index}.router.proj`,
        bits: 8,
        group_size: 64,
      })),
    };
    expect(formatInspectedQuantizationLayout(layout)).toBe(
      "4-bit group 64; 30 8-bit model.layers.N.router.proj overrides",
    );
  });

  it("keeps the Qwen3 mlp.gate map distinct from Gemma router.proj", () => {
    const layout = {
      default_bits: 4,
      default_group_size: 64,
      module_overrides: [
        { module_path: "model.layers.0.mlp.gate", bits: 8, group_size: 64 },
      ],
    };
    expect(formatInspectedQuantizationLayout(layout)).toBe(
      "4-bit group 64; 1 8-bit model.layers.N.mlp.gate override",
    );
  });
});
