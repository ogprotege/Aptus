# Training-policy bibliography

> **Documentation status:** Active research bibliography
>
> **Authority:** Non-normative citation record; does not make a method selectable
>
> **Applies to:** Training-policy increment (TP) citations, not a method catalog
>
> **Last reviewed:** 2026-08-16
>
> **Next scheduled review:** 2026-10-22, or when a listed method changes lifecycle

This file records the papers the training-policy increment consulted. Each
entry names the work, its arXiv id or journal id, what Aptus took, and what
Aptus refused. These are priors and research identities, not optima. A citation
does not prove compatibility with Aptus's pinned stack, does not expand the
method catalog, and does not authorize a quality claim from loss.

No PDF binary from the local dump is stored in this repository.

## Entries

### LoRA

- **Title:** LoRA: Low-Rank Adaptation of Large Language Models
- **Identifier:** [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- **What Aptus took:** The adapter method class already in Aptus.
- **What Aptus refused:** Treating LoRA as an optimum.

### BitFit

- **Title:** BitFit: Simple Parameter-efficient Fine-tuning for
  Transformer-based Masked Language-models
- **Identifier:** [arXiv:2106.10199](https://arxiv.org/abs/2106.10199)
- **What Aptus took:** Bibliography only.
- **What Aptus refused:** BitFit is not a selectable Aptus method.

### InstructGPT

- **Title:** Training language models to follow instructions with human
  feedback
- **Identifier:** [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- **What Aptus took:** Validation loss can fall or rise while preference does
  not equal quality.
- **What Aptus refused:** Promoting loss to quality, or to an M8 eval decision.

### Self-Instruct

- **Title:** Self-Instruct: Aligning Language Models with Self-Generated
  Instructions
- **Identifier:** [arXiv:2212.10560](https://arxiv.org/abs/2212.10560)
- **What Aptus took:** Bibliography.
- **What Aptus refused:** Synthetic-data generation in this increment.

### AdaLoRA

- **Title:** Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning
- **Identifier:** [arXiv:2303.10512](https://arxiv.org/abs/2303.10512)
- **What Aptus took:** Bibliography of the existing experimental identity.
- **What Aptus refused:** AdaLoRA is not implemented.

### LLM-Adapters

- **Title:** LLM-Adapters: An Adapter Family for Parameter-Efficient
  Fine-Tuning of Large Language Models
- **Identifier:** [arXiv:2304.01933](https://arxiv.org/abs/2304.01933)
- **What Aptus took:** Bibliography.
- **What Aptus refused:** No new adapter zoo.

### PEFT survey (Xu et al.)

- **Title:** Parameter-Efficient Fine-Tuning Methods for Pretrained Language
  Models: A Critical Review and Assessment
- **Identifier:** [arXiv:2312.12148](https://arxiv.org/abs/2312.12148)
- **What Aptus took:** Bibliography.
- **What Aptus refused:** No method-catalog expansion.

### DoRA

- **Title:** DoRA: Weight-Decomposed Low-Rank Adaptation
- **Identifier:** [arXiv:2402.09353](https://arxiv.org/abs/2402.09353)
- **What Aptus took:** Bibliography of the existing experimental identity.
- **What Aptus refused:** DoRA is not implemented.

### BiLoRA

- **Title:** BiLoRA: A Bi-level Optimization Framework for
  Overfitting-Resilient Low-Rank Adaptation of Large Pre-trained Models
- **Identifier:** [arXiv:2403.13037](https://arxiv.org/abs/2403.13037)
- **What Aptus took:** Signal only (eval-rose versus collapsed detection
  order).
- **What Aptus refused:** The method is not implemented.

### AFLoRA

- **Title:** AFLoRA: Adaptive Freezing of Low Rank Adaptation in Parameter
  Efficient Fine-Tuning of Large Models
- **Identifier:** [arXiv:2403.13269](https://arxiv.org/abs/2403.13269)
- **What Aptus took:** Bibliography of the existing research-only identity.
- **What Aptus refused:** AFLoRA is not implemented.

### PEFT survey (Han et al.)

- **Title:** Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive
  Survey
- **Identifier:** [arXiv:2403.14608](https://arxiv.org/abs/2403.14608)
- **What Aptus took:** Bibliography.
- **What Aptus refused:** No method-catalog expansion.

### ReFT

- **Title:** ReFT: Representation Finetuning for Language Models
- **Identifier:** [arXiv:2404.03592](https://arxiv.org/abs/2404.03592)
- **What Aptus took:** The existing research-only identity.
- **What Aptus refused:** ReFT stays research-only.

### ShareLoRA

- **Title:** ShareLoRA: Parameter Efficient and Robust Large Language Model
  Fine-tuning via Shared Low-Rank Adaptation
- **Identifier:** [arXiv:2406.10785](https://arxiv.org/abs/2406.10785)
- **What Aptus took:** Bibliography of the existing experimental identity.
- **What Aptus refused:** ShareLoRA is not implemented.

### Flexora

- **Title:** Flexora: Flexible Low Rank Adaptation for Large Language Models
- **Identifier:** [arXiv:2408.10774](https://arxiv.org/abs/2408.10774)
- **What Aptus took:** Signal only.
- **What Aptus refused:** The method is not implemented.

### Ultimate Guide

- **Title:** The Ultimate Guide to Fine-Tuning LLMs from Basics to
  Breakthroughs: An Exhaustive Review of Technologies, Research, Best
  Practices, Applied Research Challenges and Opportunities
- **Identifier:** [arXiv:2408.13296](https://arxiv.org/abs/2408.13296)
- **What Aptus took:** Bibliography.
- **What Aptus refused:** Its Optuna and search advice.

### 2025 NLP Journal review

- **Title:** The fine art of fine-tuning: A structured review of advanced LLM
  fine-tuning techniques
- **Identifier:** `S2949719125000202`
  ([DOI:10.1016/j.nlp.2025.100144](https://doi.org/10.1016/j.nlp.2025.100144))
- **What Aptus took:** Bibliography only.
- **What Aptus refused:** Using the review as more than a citation.

### Biderman 2024

- **Title:** LoRA Learns Less and Forgets Less
- **Identifier:** [arXiv:2405.09673](https://arxiv.org/abs/2405.09673)
- **What Aptus took:** LoRA regularizes more than weight decay.
- **What Aptus refused:** Changing `weight_decay` from 0.0, or using decay as a
  sycophancy cure.

### AdamW

- **Title:** Decoupled Weight Decay Regularization
- **Identifier:** [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
- **What Aptus took:** Citation only.
- **What Aptus refused:** No optimizer rewrite.

### QLoRA

- **Title:** QLoRA: Efficient Finetuning of Quantized LLMs
- **Identifier:** [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- **What Aptus took:** The already selectable Aptus method class.
- **What Aptus refused:** No change this increment.

### LIMA

- **Title:** LIMA: Less Is More for Alignment
- **Identifier:** [arXiv:2305.11206](https://arxiv.org/abs/2305.11206)
- **What Aptus took:** Small-corpus caution behind the 100-row supervision
  prior.
- **What Aptus refused:** Treating Path Alpha's 4-row set as real SFT.

### Sharma sycophancy

- **Title:** Towards Understanding Sycophancy in Language Models
- **Identifier:** [arXiv:2310.13548](https://arxiv.org/abs/2310.13548)
- **What Aptus took:** The parrot/sycophancy over-training prior at 10 or more
  epochs on 100–299 rows.
- **What Aptus refused:** Claiming a dataset "will produce a sycophant."

## Related documentation

- [Research and intake index](index.md)
- [Claim language](../product/claim-language.md)
- [Fine-tuning method taxonomy](../methodology/method-taxonomy.md)
- [Training policy and run-correction implementation plan](../superpowers/plans/2026-08-16-training-policy-and-run-correction.md)
- [EXAMPLE forensic review and salvage ledger](example-intake-reconciliation.md)
