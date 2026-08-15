/** Presentation-only mapping of free-text rejection_reasons (mirrors aptus.refusal). */

export interface RefusalGuidance {
  reasonCode: string;
  title: string;
  explanation: string;
  changeableFacts: string[];
  operatorActionable: boolean;
  noneInCatalog: boolean;
  sourceReason: string;
}

const NONE_IN_CATALOG =
  "No supported correction exists in the current Aptus catalog for these facts.";

type Rule = {
  needle: string;
  reasonCode: string;
  title: string;
  explanation: string;
  changeableFacts: string[];
  operatorActionable: boolean;
  noneInCatalog: boolean;
};

const RULES: Rule[] = [
  {
    needle: "full-parameter fp16",
    reasonCode: "full_fp16",
    title: "Full FP16 training is closed",
    explanation:
      "Full-parameter training requires BF16 on every participating device. The FP16 full path is fail-closed.",
    changeableFacts: ["hardware.devices[].supports_bf16", "method"],
    operatorActionable: true,
    noneInCatalog: false,
  },
  {
    needle: "full-parameter fsdp",
    reasonCode: "full_fsdp",
    title: "Full FSDP is closed",
    explanation: "Full-parameter FSDP is outside the verified v0.2 matrix.",
    changeableFacts: ["distribution", "method"],
    operatorActionable: true,
    noneInCatalog: false,
  },
  {
    needle: "with fsdp is outside the verified",
    reasonCode: "quantized_fsdp",
    title: "Quantized FSDP is closed",
    explanation: "int8-LoRA and QLoRA with FSDP are outside the verified matrix.",
    changeableFacts: ["distribution", "method"],
    operatorActionable: true,
    noneInCatalog: false,
  },
  {
    needle: "requires at least two gpus",
    reasonCode: "multi_gpu_on_single",
    title: "Multi-GPU placement needs at least two devices",
    explanation:
      "DDP and FSDP stay visible on a single-GPU inventory but are unsupported. Planner support is not multi-GPU runtime proof.",
    changeableFacts: ["hardware.devices", "distribution"],
    operatorActionable: true,
    noneInCatalog: false,
  },
  {
    needle: "has no registered full compiler",
    reasonCode: "mlx_full",
    title: "Full fine-tuning is not compiled for this runtime",
    explanation: "Use LoRA/QLoRA on MLX, or full training on CUDA hardware.",
    changeableFacts: ["method", "training_runtime", "hardware.backend"],
    operatorActionable: true,
    noneInCatalog: false,
  },
  {
    needle: "usable per-device memory is unknown",
    reasonCode: "unknown_device_free_memory",
    title: "Free device memory is unknown",
    explanation:
      "Aptus will not treat total VRAM or total unified memory as free. Measure or declare current free memory.",
    changeableFacts: ["hardware.devices[].free_vram_bytes", "hardware.host_ram_free_bytes"],
    operatorActionable: true,
    noneInCatalog: false,
  },
  {
    needle: "host ram free is unknown",
    reasonCode: "unknown_host_ram_free",
    title: "Free host RAM is unknown",
    explanation: "Aptus will not treat total host RAM as free. Measure or declare current host RAM free.",
    changeableFacts: ["hardware.host_ram_free_bytes"],
    operatorActionable: true,
    noneInCatalog: false,
  },
  {
    needle: "free disk is unknown",
    reasonCode: "unknown_disk_free",
    title: "Free disk is unknown",
    explanation: "Aptus will not assume enough staging space when free disk is omitted.",
    changeableFacts: ["hardware.disk_free_bytes"],
    operatorActionable: true,
    noneInCatalog: false,
  },
  {
    needle: "will not invent 4",
    reasonCode: "missing_intermediate_size",
    title: "MLP adapter width is unknown",
    explanation: "intermediate_size is required for MLP adapter targets. Aptus will not invent 4 × hidden_size.",
    changeableFacts: ["model.intermediate_size"],
    operatorActionable: true,
    noneInCatalog: false,
  },
  {
    needle: "even the point estimate exceeds",
    reasonCode: "infeasible_memory",
    title: "Estimated memory exceeds usable device capacity",
    explanation: "The analytic point estimate already exceeds usable per-device memory after reserve.",
    changeableFacts: [
      "target.sequence_length",
      "target.effective_batch_size",
      "method",
      "hardware free memory",
    ],
    operatorActionable: true,
    noneInCatalog: false,
  },
  {
    needle: "upper envelope exceeds usable",
    reasonCode: "conditional_upper_envelope",
    title: "Upper memory envelope exceeds capacity",
    explanation:
      "The point estimate fits, but the uncalibrated upper envelope does not. Pilot evidence is still required.",
    changeableFacts: ["target.sequence_length", "target.effective_batch_size", "method"],
    operatorActionable: true,
    noneInCatalog: false,
  },
  {
    needle: "mlx-lm support is pilot-required",
    reasonCode: "conditional_pilot_required",
    title: "MLX-LM path is pilot-required",
    explanation:
      "Unified-memory estimates are provisional. A real-model pilot must pass before confirmed full-duration training.",
    changeableFacts: [],
    operatorActionable: false,
    noneInCatalog: true,
  },
  {
    needle: "fsdp uses a simplified uncalibrated",
    reasonCode: "conditional_fsdp_pilot",
    title: "LoRA FSDP requires a real-model pilot",
    explanation: "FSDP sharding priors are uncalibrated. Analytic fit is not multi-rank proof.",
    changeableFacts: [],
    operatorActionable: false,
    noneInCatalog: true,
  },
  {
    needle: "sequence length exceeds the model context",
    reasonCode: "sequence_length_exceeds_context",
    title: "Sequence length exceeds model context",
    explanation: "Lower sequence length or use a longer-context model revision.",
    changeableFacts: ["target.sequence_length", "model.context_length"],
    operatorActionable: true,
    noneInCatalog: false,
  },
];

export function guideRejectionReason(reason: string): RefusalGuidance {
  const text = reason.trim();
  if (!text) {
    return {
      reasonCode: "empty_reason",
      title: "Unspecified refusal",
      explanation: "The planner returned an empty rejection reason.",
      changeableFacts: [],
      operatorActionable: false,
      noneInCatalog: true,
      sourceReason: reason,
    };
  }
  const lowered = text.toLowerCase();
  for (const rule of RULES) {
    if (lowered.includes(rule.needle)) {
      return {
        reasonCode: rule.reasonCode,
        title: rule.title,
        explanation: rule.explanation,
        changeableFacts: rule.noneInCatalog ? [] : rule.changeableFacts,
        operatorActionable: rule.operatorActionable,
        noneInCatalog: rule.noneInCatalog,
        sourceReason: text,
      };
    }
  }
  return {
    reasonCode: "unclassified_reason",
    title: "Planner refused this path",
    explanation: text,
    changeableFacts: [],
    operatorActionable: false,
    noneInCatalog: true,
    sourceReason: text,
  };
}

export function whatCanChange(guidance: RefusalGuidance): string {
  if (guidance.noneInCatalog || guidance.changeableFacts.length === 0) {
    return NONE_IN_CATALOG;
  }
  return guidance.changeableFacts.join(", ");
}

export function fitStatusLabel(status: string): string {
  const normalized = status.toLowerCase();
  if (normalized === "conditional") {
    return "conditional · pilot required";
  }
  if (normalized === "unsupported") {
    return "unsupported · not runtime-ready";
  }
  if (normalized === "infeasible") {
    return "infeasible · hard resource gate";
  }
  if (normalized === "feasible") {
    return "feasible · still evidence-gated at run time";
  }
  return status.replace(/-/g, " ");
}
