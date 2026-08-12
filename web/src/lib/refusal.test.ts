import { describe, expect, it } from "vitest";
import { fitStatusLabel, guideRejectionReason, whatCanChange } from "./refusal";

describe("refusal guidance", () => {
  it("maps full FP16 free text to changeable facts", () => {
    const guided = guideRejectionReason(
      "Full-parameter FP16 training is fail-closed in Aptus v0.2 because the generated mixed-precision path does not retain verified FP32 trainable master weights.",
    );
    expect(guided.reasonCode).toBe("full_fp16");
    expect(guided.changeableFacts).toContain("hardware.devices[].supports_bf16");
    expect(whatCanChange(guided)).toContain("supports_bf16");
  });

  it("keeps multi-GPU single-device rows from reading as ready", () => {
    const guided = guideRejectionReason("ddp requires at least two GPUs.");
    expect(guided.reasonCode).toBe("multi_gpu_on_single");
    expect(fitStatusLabel("unsupported")).toContain("not runtime-ready");
  });

  it("states none-in-catalog for MLX pilot-required strings", () => {
    const guided = guideRejectionReason(
      "MLX-LM support is pilot-required: the unified-memory estimate is provisional and cannot guarantee that the exact model and data fit.",
    );
    expect(guided.noneInCatalog).toBe(true);
    expect(whatCanChange(guided)).toMatch(/No supported correction exists/);
  });

  it("labels conditional as pilot required, not success", () => {
    expect(fitStatusLabel("conditional")).toBe("conditional · pilot required");
    expect(fitStatusLabel("feasible")).not.toMatch(/success/i);
  });
});
