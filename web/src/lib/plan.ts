import type {
  BatchStrategy,
  CandidatePlan,
  MemoryComponent,
  MemoryComponentValue,
  TrainingPlan,
} from "../types";

const COMPONENT_LABELS: Record<string, string> = {
  base_weights: "Base weights",
  quantization_metadata: "Quantization metadata",
  adapter_weights: "Adapter weights",
  adapter_gradients: "Adapter gradients",
  optimizer_states: "Optimizer states",
  activations: "Activations",
  temporary_overhead: "Temporary overhead",
  safety_margin: "Safety margin",
  user_reserve: "User reserve",
};

function titleCase(value: string): string {
  return value
    .replace(/_bytes$/, "")
    .split(/[_-]/)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

export function formatMethod(method: string): string {
  const known: Record<string, string> = {
    lora: "LoRA",
    qlora: "QLoRA",
    "int8-lora": "8-bit LoRA",
    full: "Full fine-tuning",
  };
  return known[method.toLowerCase()] ?? method;
}

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined || !Number.isFinite(bytes)) {
    return "Not measured";
  }
  return `${(bytes / 1024 ** 3).toFixed(bytes >= 10 * 1024 ** 3 ? 1 : 2)} GiB`;
}

export function candidateStatus(candidate: CandidatePlan): string {
  if (candidate.status) return candidate.status;
  if (candidate.feasible === true) return "feasible";
  if (candidate.feasible === false) return "infeasible";
  return "unknown";
}

export function candidateBatches(candidate: CandidatePlan): BatchStrategy {
  return {
    micro_batch_size:
      candidate.batches?.micro_batch_size ?? candidate.micro_batch_size,
    gradient_accumulation_steps:
      candidate.batches?.gradient_accumulation_steps ??
      candidate.gradient_accumulation_steps,
    effective_batch_size:
      candidate.batches?.effective_batch_size ?? candidate.effective_batch_size,
  };
}

export function memoryComponents(candidate: CandidatePlan | null): MemoryComponent[] {
  if (!candidate?.memory) return [];
  const supplied = candidate.memory.components;
  if (Array.isArray(supplied)) {
    return supplied.filter((component) => component.expected_bytes > 0);
  }
  if (supplied && typeof supplied === "object") {
    return Object.entries(supplied)
      .map(([key, value]) => {
        if (typeof value === "number") {
          return {
            key,
            label: COMPONENT_LABELS[key] ?? titleCase(key),
            expected_bytes: value,
          };
        }
        const typed = value as MemoryComponentValue;
        return {
          key,
          label: typed.label ?? COMPONENT_LABELS[key] ?? titleCase(key),
          expected_bytes: typed.expected_bytes ?? typed.bytes ?? 0,
          upper_bytes: typed.upper_bytes,
        };
      })
      .filter((component) => component.expected_bytes > 0);
  }

  const upperBounds = candidate.memory.component_upper_bounds;
  const pointComponents = Object.entries(candidate.memory)
    .filter(
      ([key, value]) =>
        key.endsWith("_bytes") &&
        !["expected_bytes", "upper_bytes", "estimated_peak_bytes", "point_estimate_bytes", "upper_estimate_bytes", "uncertainty_bytes", "limit_bytes", "device_total_bytes"].includes(key) &&
        !(upperBounds && key === "safety_margin_bytes") &&
        typeof value === "number" &&
        value > 0,
    )
    .map(([key, value]) => {
      const cleanKey = key.replace(/_bytes$/, "");
      return {
        key: cleanKey,
        label: COMPONENT_LABELS[cleanKey] ?? titleCase(cleanKey),
        expected_bytes: value as number,
        upper_bytes: upperBounds?.[key],
      };
    });
  if (upperBounds?.uncertainty_bytes) {
    pointComponents.push({
      key: "uncertainty",
      label: "Uncertainty",
      expected_bytes: 0,
      upper_bytes: upperBounds.uncertainty_bytes,
    });
  }
  return pointComponents;
}

export function expectedMemory(candidate: CandidatePlan | null): number | null {
  if (!candidate?.memory) return null;
  return (
    candidate.memory.expected_bytes ??
    candidate.memory.point_estimate_bytes ??
    candidate.memory.estimated_peak_bytes ??
    memoryComponents(candidate).reduce(
      (total, component) => total + component.expected_bytes,
      0,
    ) ??
    null
  );
}

export function upperMemory(candidate: CandidatePlan | null): number | null {
  if (!candidate?.memory) return null;
  return candidate.memory.upper_bytes ?? candidate.memory.upper_estimate_bytes ?? expectedMemory(candidate);
}

export function memoryLimit(candidate: CandidatePlan | null): number | null {
  return candidate?.memory?.limit_bytes ?? null;
}

export function planRationale(plan: TrainingPlan): string[] {
  return plan.rationale.length
    ? plan.rationale
    : plan.recommendation_rationale ?? [];
}

export function isPassingValidation(state: string | undefined): boolean {
  return ["contract-pass", "static-pass", "dependency-pass", "model-data-pass", "measured-preflight-pass", "pilot-pass", "execution-approved", "measured-run-pass"].includes(
    state ?? "",
  );
}

const VALIDATION_RANK: Record<string, number> = {
  "contract-pass": 1,
  "static-pass": 2,
  "dependency-pass": 3,
  "model-data-pass": 4,
  "measured-preflight-pass": 5,
  "pilot-pass": 6,
  "execution-approved": 7,
  "measured-run-pass": 8,
};

export function canStartAction(
  state: string | undefined,
  action: "dependency" | "model-data" | "preflight" | "pilot" | "train",
): boolean {
  const rank = VALIDATION_RANK[state ?? ""] ?? 0;
  const requiredRank = {
    dependency: 2,
    "model-data": 3,
    preflight: 4,
    pilot: 5,
    train: 6,
  }[action];
  return rank >= requiredRank;
}

export function nextForwardAction(
  state: string | undefined,
): "dependency" | "model-data" | "preflight" | "pilot" | "train" {
  const rank = VALIDATION_RANK[state ?? ""] ?? 0;
  if (rank < 3) return "dependency";
  if (rank < 4) return "model-data";
  if (rank < 5) return "preflight";
  if (rank < 6) return "pilot";
  return "train";
}
