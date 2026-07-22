
# MCP Hyperparameter Optimization Method Catalog

> **Documentation status:** Archived research intake
>
> **Authority:** Non-normative. Its uncited ranges and formulas are not Aptus
> planner defaults.
>
> **Last reviewed:** 2026-07-22
>
> **Next scheduled review:** 2027-07-22, or when its archive disposition changes
>
> Use the [method taxonomy](../docs/methodology/method-taxonomy.md) for current
> classification. See the
> [reconciliation ledger](../docs/research/reference-and-to-review-reconciliation.md#referencehparam_methods_referencemd)
> for accepted concepts and rejected numeric claims.

This file contains extracted and structured tuning methods from key academic papers. Each method includes a description, use case, and any known formulas or configuration heuristics.

---

## 1. LoRA (Low-Rank Adaptation)

**Description:** Injects low-rank matrices A and B into the weight update:  
ΔW = A · B, where A ∈ ℝ^(d×r), B ∈ ℝ^(r×k)

**Typical Config:**
- Rank (r): 4–16
- Alpha: 16–64
- Dropout: 0.0–0.1

**Equation Heuristic:**  
Learning Rate ~ 2e-4 (larger r → smaller LR)  
Batch Size dependent on available VRAM  
Recommended: AdamW + Linear Warmup

**Use Case:** Efficient full-model adaptation with minimal added parameters.

---

## 2. QLoRA (Quantized LoRA)

**Description:** Applies 4-bit quantization (NF4) to base model and runs LoRA over it. Uses paged optimizers to fit within VRAM.

**Typical Config:**
- Quantization: 4-bit NF4
- LoRA Rank: 8
- LoRA Alpha: 32
- Optimizer: PagedAdamW8bit

**Memory Heuristic:** Enables fine-tuning of 65B model on 48–72GB GPU

**Use Case:** Fine-tuning large models under memory constraints.

---

## 3. AdaLoRA (Adaptive LoRA)

**Description:** Dynamically allocates LoRA rank during training via importance scoring.

**Formula:**  
Dynamic allocation of rank rᵢ ∝ ||∂L/∂Wᵢ||₂  
Weighted based on Fisher importance scores.

**Use Case:** Balances expressiveness and efficiency in varied training phases.

---

## 4. BiLoRA (Bi-level LoRA)

**Description:** Optimizes singular values and directions on disjoint data partitions to reduce overfitting.

**Config:**
- SV and direction updates split by data shard
- γ₁ regularizer applied to orthogonality

**Heuristic:** Less sensitive to hyperparameters than LoRA; fewer tuning steps needed.

---

## 5. ShareLoRA

**Description:** Shares adaptation weights (A or B) across layers to reduce parameter count.

**Variants:**
- Share-A
- Share-B
- Share-AB

**Performance Tip:** Share-A often performs best; avoids full weight redundancy.

---

## 6. BitFit

**Description:** Fine-tunes only bias terms in transformer layers.

**Config:**
- Only bias tensors are set as trainable (e.g., `linear.bias`)

**Use Case:** Ultra-light adaptation with minimal computation.

---

## 7. Intrinsic SAID

**Description:** Decomposes model into structural vs informational components.

**Use Case:** Interpretability and efficient targeted fine-tuning.

---

## 8. SAM (Sharpness-Aware Minimization)

**Description:** Optimizes for flat minima to enhance generalization.

**Update Rule:**  
θ ← θ - η ∇L(θ + ρ ∇L(θ)/||∇L(θ)||)

**Config:**
- ρ (perturbation scale): 0.05–0.3

**Use Case:** Prevents sharp loss spikes and improves robustness.

---
