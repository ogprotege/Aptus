# The Top 50 Algorithmic Methods for Adapting, Fine-Tuning, Aligning, and Training Large Language Models

> **Documentation status:** Active research-only source
>
> **Authority:** Non-normative. A listed method or library mapping is not an
> Aptus capability.
>
> **Last reviewed:** 2026-07-22
>
> **Next scheduled review:** 2026-10-22, or when the research cutoff changes
>
> Read the [Reference packet boundary](README.md) and
> [reconciliation ledger](../docs/research/reference-and-to-review-reconciliation.md)
> before using this material. Revalidate library support against Aptus's pinned
> runtime before implementation.

A source-grounded technical reference. Every factual claim links to a primary paper, official documentation, or authoritative repository fetched during research. Methods are algorithms, objectives, or parameterizations implementable in a Python script with PyTorch or JAX plus open-source libraries; libraries are treated as implementation aids, not as methods.

Research cutoff: July 21, 2026. Library-support claims reflect Hugging Face TRL, PEFT (through v0.17.0 plus the current main branch), and mlx-lm documentation as fetched on that date.

---

## Executive Summary

This report ranks and documents 50 genuinely distinct algorithmic methods for changing a large language model's weights or behavior through training, from the least invasive (freezing the entire network and learning a handful of soft-prompt vectors) to the most invasive (compute-optimal pretraining from scratch). The intervention spectrum is organized into nine families and six tiers.

The dominant practical pattern in 2026 remains a layered stack: a base model is pretrained with a causal-LM objective, optionally continued-pretrained on domain data, adapted to instructions via supervised fine-tuning (SFT), and aligned to preferences via a direct method such as DPO ([Rafailov et al., DPO](https://arxiv.org/abs/2305.18290)) or an online RL method such as GRPO ([Shao et al., DeepSeekMath](https://arxiv.org/abs/2402.03300)), with parameter-efficient methods, above all LoRA ([Hu et al., LoRA](https://arxiv.org/abs/2106.09685)) and its quantized recipe QLoRA ([Dettmers et al., QLoRA](https://arxiv.org/abs/2305.14314)), used at any stage to cut the memory bill.

The most important development since mid-2024 is the consolidation of critic-free, group-relative RL as the default reasoning-alignment path, and the rapid proliferation of GRPO successors. DAPO ([Yu et al., DAPO](https://arxiv.org/abs/2503.14476)), Dr. GRPO ([Liu et al., Understanding R1-Zero](https://arxiv.org/abs/2503.20783)), GSPO ([Zheng et al., GSPO](https://arxiv.org/abs/2507.18071)), REINFORCE++ ([Hu et al., REINFORCE++](https://arxiv.org/abs/2501.03262)), and CISPO ([MiniMax, MiniMax-M1](https://arxiv.org/abs/2506.13585)) all appeared in 2025 and are now selectable inside a single TRL `GRPOConfig` through `loss_type` and `importance_sampling_level` flags ([TRL GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer)). On the parameter-efficient side, the PEFT library added a wave of 2025 methods including MiSS (which deprecates Bone), RandLoRA, SHiRA, RoAd, and C3A ([PEFT PEFT types](https://huggingface.co/docs/peft/v0.17.0/en/package_reference/peft_types)).

Ranking rule. Ranks 1 to 50 reflect a composite judgment across six axes: practical utility in 2026, maturity and reproducibility, breadth of use, compute efficiency, quality potential, and conceptual distinctness. They are not a claim of universally superior benchmark performance (the KTO authors explicitly note that "there is no one HALO that is universally superior" ([Ethayarajh et al., KTO](https://arxiv.org/abs/2402.01306))). A method that is ubiquitous, well-supported, and reproducible (such as LoRA, SFT, or DPO) outranks a newer, more parameter-efficient method (such as SHiRA or ReFT) even where the newer method reports better parameter counts. Each entry carries an intervention tier (T1 to T6, least to most invasive) and a Type label so readers can navigate by how much of the model they change and by what kind of object each method actually is.

Type vocabulary. To avoid the common category error of listing an objective, a pipeline, and a parameterization as if they were the same kind of thing, every entry is tagged as one of:
- Objective: a loss function or training target (for example DPO, KD, causal-LM pretraining).
- Parameterization: a choice of which parameters exist and are trained, given some objective (for example LoRA, DoRA, prefix-tuning, BitFit).
- Recipe: a named composition of an objective, a parameterization, and infrastructure (for example QLoRA).
- Pipeline: an outer generate-filter-retrain loop wrapped around one or more inner objectives (for example RLAIF, RAFT, STaR, Self-Instruct, SPIN).
- Modifier: a train-time perturbation or regularizer added on top of another objective (for example NEFTune, EWC, LwF, replay).
- Full-parameter training strategy: a method that trains all weights but restructures the gradient or optimizer footprint to make that feasible (for example GaLore, LOMO, MeZO, LISA). These are retained, unlike bare optimizer swaps, because they change what is representable or storable during the update, not merely the numeric update rule; see the exclusions section for why Sophia and Adam-mini are not ranked.

Critical caveat, stated up front. Several widely cited efficiency techniques are not fine-tuning methods and are excluded from the ranking: gradient checkpointing, mixed precision, ZeRO and FSDP, tensor and pipeline parallelism, FlashAttention, sequence packing, and optimizer offload. These are infrastructure enablers that make a given method cheaper; they do not define a distinct training objective or parameterization. They are treated in a dedicated warning section.

---

## Taxonomy and Intervention Tiers

| Tier | Name | What changes | Representative methods |
|---|---|---|---|
| T1 | Input / representation-level | No base weights; learn prompt embeddings or hidden-state interventions | Prompt Tuning, Prefix-Tuning, P-Tuning v2, LoReFT |
| T2 | Additive / injected PEFT | Small added modules; base frozen | Adapters, LoRA, DoRA, VeRA, IA3, AdaLoRA, LoRA+, rsLoRA, MiSS, BOFT/OFT |
| T3 | Sparse / structural PEFT | A pre-existing subset of base weights | BitFit (biases), LISA (sampled layers) |
| T4 | Quantized low-rank recipes | Frozen quantized base + trainable low-rank | QLoRA, LoftQ, PiSSA/QPiSSA |
| T5 | Full-parameter fine-tuning and alignment | All (or nearly all) weights updated | SFT, Instruction Tuning, DAPT/TAPT, RLHF/PPO, DPO, KTO, ORPO, SimPO, GRPO, DAPO, GSPO, Dr. GRPO, IPO, RLOO, RLAIF, RAFT, SPIN, STaR, KD, GKD, MiniLLM, Self-Instruct, GaLore, LOMO, MeZO, EWC, LwF, Replay, NEFTune |
| T6 | From-scratch / continued pretraining | Full weights, corpus-scale objective | Causal-LM pretraining (Chinchilla-optimal), Continued Pretraining |

The nine functional families requested map onto these tiers as: (1) soft-prompt/representation to T1; (2) adapters/PEFT to T2/T3; (3) low-rank and quantized to T2/T4; (4) supervised/instruction/domain to T5/T6; (5) preference/RL to T5; (6) continual/anti-forgetting to T5; (7) memory-efficient full-parameter to T5; (8) distillation/self-training to T5; (9) full/from-scratch pretraining to T6.

---

## Ranked Methods 1 to 50

Compute/VRAM profile scale: Very low / Low / Medium / High / Extreme, relative to full fine-tuning of the same model. In the implementation column, "Native" means a first-class class or flag in an open-source library (with experimental status flagged where the docs say so); "Custom" means implement from the paper's loss or primitive.

### Core methods most teams run in 2026 (ranks 1 to 17)

| # | Method (acronym) | Type | Family / Tier | Parameter scope | Core mechanism (one sentence) | Best use cases | Compute / VRAM | Python implementation mapping | Principal limitation | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Supervised Fine-Tuning (SFT) | Objective | Supervised adaptation / T5 | All weights, or combine with any PEFT | Minimize token-level cross-entropy of target completions given prompts, optionally masking the prompt so loss is computed on completion or assistant tokens only ([TRL SFT Trainer](https://huggingface.co/docs/trl/sft_trainer)). | Teaching formats, tasks, and domains; the base layer under almost every alignment pipeline. | High (full); Low with PEFT | Native: `trl.SFTTrainer` + `SFTConfig` (`completion_only_loss`, `assistant_only_loss`, `packing`) ([TRL SFT Trainer](https://huggingface.co/docs/trl/sft_trainer)); or plain HF `Trainer` with causal-LM loss. | Can overfit or forget; quality bounded by data quality. | established |
| 2 | LoRA, Low-Rank Adaptation | Parameterization | Low-rank PEFT / T2 | Frozen base + injected trainable rank-decomposition matrices per layer | Freeze pretrained weights and inject trainable low-rank matrices (delta W = BA) into Transformer layers, reducing trainable params by up to about 10,000x and GPU memory about 3x versus full Adam fine-tuning ([Hu et al., LoRA](https://arxiv.org/abs/2106.09685)). | Default PEFT for single-task and multi-adapter deployment; mergeable, no added inference latency ([Hu et al., LoRA](https://arxiv.org/abs/2106.09685)). | Low / Low | Native: `peft.LoraConfig` + `get_peft_model` ([PEFT LoRA reference](https://huggingface.co/docs/peft/package_reference/lora)). | Confines updates to a low-rank subspace, sometimes below full FT ([Zhao et al., GaLore](https://arxiv.org/abs/2403.03507)). | established |
| 3 | QLoRA | Recipe | Quantized low-rank recipe / T4 | Frozen 4-bit base + trainable LoRA adapters | Backpropagate through a frozen 4-bit NF4-quantized base into LoRA adapters, adding double quantization and paged optimizers to finetune a 65B model on one 48GB GPU at 16-bit-equivalent quality ([Dettmers et al., QLoRA](https://arxiv.org/abs/2305.14314)). | Large-model fine-tuning on one consumer or prosumer GPU. Honestly a recipe: 4-bit quantization + LoRA + paged optimizers, but established enough to list. | Low / Very low (per model size) | Native: `bitsandbytes` 4-bit load + `peft.LoraConfig` ([Dettmers et al., QLoRA](https://arxiv.org/abs/2305.14314); [PEFT LoRA reference](https://huggingface.co/docs/peft/package_reference/lora)). | Quantization error can gap versus full FT; slower per step than bf16 LoRA. | established |
| 4 | DPO, Direct Preference Optimization | Objective | Preference optimization / T5 | Policy weights (full or +PEFT) | Reparameterize the RLHF reward so the optimal policy is closed-form, turning alignment into a simple classification loss on preferred and dispreferred pairs with no sampling or RL loop ([Rafailov et al., DPO](https://arxiv.org/abs/2305.18290)). | Offline preference alignment when pairwise data exists; the default post-SFT alignment method. | Medium / Medium (needs reference model) | Native: `trl.DPOTrainer` ([TRL index](https://huggingface.co/docs/trl/index)). | Relies on the pairwise-to-pointwise approximation and can overfit deterministic preferences ([Azar et al., IPO](https://arxiv.org/abs/2310.12036)). | established |
| 5 | Instruction Tuning (FLAN-style) | Objective | Multitask supervised / T5 | All weights, or +PEFT | Fine-tune on many NLP tasks verbalized as natural-language instructions to unlock zero-shot generalization to unseen task types ([Wei et al., FLAN](https://arxiv.org/abs/2109.01652)). | Turning a base model into a general instruction-follower. | High / Medium-High | Native: `trl.SFTTrainer` over instruction-formatted data ([TRL SFT Trainer](https://huggingface.co/docs/trl/sft_trainer)). | Gains depend on number of datasets, scale, and template quality ([Wei et al., FLAN](https://arxiv.org/abs/2109.01652)). | established |
| 6 | GRPO, Group Relative Policy Optimization | Objective | RL alignment / T5 | Policy weights (no value network) | A PPO variant that replaces the learned critic with group-relative advantages computed from multiple sampled completions, cutting PPO memory while improving reasoning ([Shao et al., DeepSeekMath](https://arxiv.org/abs/2402.03300)). | Reasoning and verifiable-reward RL (math, code) at lower memory than PPO. | High / Medium-High | Native: `trl.GRPOTrainer` with vLLM sampling ([TRL GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer)). | Still online RL: reward design and sampling cost dominate; has a documented length and difficulty bias later addressed by Dr. GRPO ([Liu et al., Understanding R1-Zero](https://arxiv.org/abs/2503.20783)). | established |
| 7 | RLHF with PPO | Objective + pipeline | RL alignment / T5 | Policy (+ reward model + value head) | Collect ranked outputs, train a reward model, then optimize the SFT policy with reinforcement learning to maximize reward without drifting too far from the reference ([Ouyang et al., InstructGPT](https://arxiv.org/abs/2203.02155)); PPO supplies the clipped surrogate policy-gradient update ([Schulman et al., PPO](https://arxiv.org/abs/1707.06347)). | Highest-control alignment when a reward model and online sampling are available. | Extreme / High (four models in memory) | Native: `trl.PPOTrainer` (marked experimental in current docs) + `trl.RewardTrainer` ([TRL index](https://huggingface.co/docs/trl/index)). | Complex, unstable, hyperparameter-sensitive ([Rafailov et al., DPO](https://arxiv.org/abs/2305.18290); [Ahmadian et al., RLOO](https://arxiv.org/abs/2402.14740)). | established |
| 8 | DAPO, Decoupled Clip and Dynamic Sampling Policy Optimization | Objective | RL alignment / T5 | Policy weights (no value network) | A GRPO-family objective combining four techniques: Clip-Higher (to avoid entropy collapse), Dynamic Sampling, Token-Level Policy Gradient Loss, and Overlong Reward Shaping, while excluding the KL term ([Yu et al., DAPO](https://arxiv.org/abs/2503.14476)). | Large-scale, long-chain-of-thought reasoning RL where GRPO collapses or wastes tokens. | High / Medium-High | Native: `trl.GRPOConfig(loss_type="dapo", ...)` (DAPO is the current default `loss_type`), with `delta` for two-sided (Clip-Higher) clipping and `mask_truncated_completions` ([TRL GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer)). | Reference RL system built on the verl framework, not TRL; several components are separate flags rather than one switch ([Yu et al., DAPO](https://arxiv.org/abs/2503.14476)). | emerging |
| 9 | Continued / Continual Pretraining (CPT) | Objective | Full continued pretraining / T6 | All weights | Resume causal-LM training of an existing base model on new large corpora (for example 120B math tokens) to inject domain knowledge before task adaptation ([Shao et al., DeepSeekMath](https://arxiv.org/abs/2402.03300)). | Domain or language injection at scale before SFT. | Extreme / High | Custom or Native: HF `Trainer` with a causal-LM objective on the domain corpus; scale per compute-optimal law ([Hoffmann et al., Chinchilla](https://arxiv.org/abs/2203.15556)). | Catastrophic forgetting of general ability; expensive. | established |
| 10 | DoRA, Weight-Decomposed Low-Rank Adaptation | Parameterization | Low-rank PEFT / T2 | Frozen base + trainable direction (LoRA) and magnitude | Decompose each pretrained weight into magnitude and direction, applying LoRA only to the direction to recover more of full-FT learning capacity with no added inference cost ([Liu et al., DoRA](https://arxiv.org/abs/2402.09353)). | Drop-in LoRA upgrade when the LoRA-to-full-FT accuracy gap matters, especially at low rank. | Low / Low | Native: `peft.LoraConfig(use_dora=True)` ([PEFT LoRA reference](https://huggingface.co/docs/peft/package_reference/lora)); also MLX `--fine-tune-type dora` ([mlx-lm LORA docs](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)). | Extra overhead versus pure LoRA; linear and Conv layers only ([PEFT LoRA reference](https://huggingface.co/docs/peft/package_reference/lora)). | established |
| 11 | KTO, Kahneman-Tversky Optimization | Objective | Preference optimization / T5 | Policy weights (+ reference) | Maximize a prospect-theory human-utility objective (a HALO) from a binary desirable or undesirable signal, avoiding the need for paired preferences ([Ethayarajh et al., KTO](https://arxiv.org/abs/2402.01306)). | Alignment when only unpaired thumbs-up or thumbs-down labels exist. | Medium / Medium | Native: `trl.KTOTrainer` (docs note KTO is now stable) ([TRL index](https://huggingface.co/docs/trl/index)). | Best loss is setting-dependent; no universally superior HALO ([Ethayarajh et al., KTO](https://arxiv.org/abs/2402.01306)). | established |
| 12 | ORPO, Odds Ratio Preference Optimization | Objective | Preference optimization / T5 | Policy weights (reference-free) | Fold preference alignment into SFT via a monolithic odds-ratio penalty on the disfavored style, removing the separate alignment phase and the reference model ([Hong et al., ORPO](https://arxiv.org/abs/2403.07691)). | One-stage SFT plus alignment on a single preference dataset. | Medium / Low-Medium (no ref model) | Native: `trl.ORPOTrainer` (marked experimental in current docs) ([TRL index](https://huggingface.co/docs/trl/index)). | Couples SFT and preference dynamics; newer than DPO. | established |
| 13 | SimPO, Simple Preference Optimization | Objective | Preference optimization / T5 | Policy weights (reference-free) | Use average per-token log-probability as a length-normalized implicit reward with a target margin, eliminating the reference model for a simpler, memory-lighter DPO alternative ([Meng et al., SimPO](https://arxiv.org/abs/2405.14734)). | Reference-free alignment where memory or latency matter. | Medium / Low-Medium | Native: `trl.CPOTrainer(loss_type="simpo", cpo_alpha=...)`; the SimPO loss is exposed through the (experimental) CPO trainer rather than a standalone SimPO trainer ([TRL index](https://huggingface.co/docs/trl/index)). | Sensitive to margin and length hyperparameters. | emerging |
| 14 | GSPO, Group Sequence Policy Optimization | Objective | RL alignment / T5 | Policy weights (no value network) | Define the importance ratio at the sequence level (from sequence likelihood) and perform sequence-level clipping, rewarding, and optimization, which stabilizes MoE RL and contributed to the Qwen3 models ([Zheng et al., GSPO](https://arxiv.org/abs/2507.18071)). | Stable large-scale and Mixture-of-Experts RL where token-level ratios (GRPO) destabilize training. | High / Medium-High | Native: `trl.GRPOConfig(importance_sampling_level="sequence")`, which the TRL docs attribute to the GSPO paper ([TRL GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer)). | Very new (July 2025); a configuration of GRPO rather than a separate trainer. | emerging |
| 15 | GaLore, Gradient Low-Rank Projection | Full-parameter training strategy | Memory-efficient full-parameter / T5 | All weights (low-rank gradient and optimizer subspace) | Project gradients into a periodically-updated low-rank subspace to shrink optimizer state by up to 65.5% while still doing full-parameter learning, enabling 7B pretraining on a 24GB RTX 4090 ([Zhao et al., GaLore](https://arxiv.org/abs/2403.03507)). | Full-parameter FT or pretraining on limited VRAM without LoRA's low-rank weight constraint. | High / Low-Medium | Native: GaLore optimizer integrated in HF `Trainer` (`optim="galore_adamw"`) ([Zhao et al., GaLore](https://arxiv.org/abs/2403.03507)). | Projection overhead; subspace-refresh hyperparameters. | emerging |
| 16 | AdaLoRA, Adaptive Budget Allocation | Parameterization | Low-rank PEFT / T2 | Frozen base + SVD-parameterized adapters with adaptive rank | Parameterize LoRA updates as SVD and prune unimportant singular values so the rank budget is allocated by importance across matrices ([Zhang et al., AdaLoRA](https://arxiv.org/abs/2303.10512)). | Fixed PEFT budget where some layers deserve more rank; low-budget regimes. | Low / Low | Native: `peft.AdaLoraConfig` ([PEFT PEFT types](https://huggingface.co/docs/peft/v0.17.0/en/package_reference/peft_types)). | Extra importance-scoring and pruning machinery versus LoRA. | established |
| 17 | Dr. GRPO, Group Relative Policy Optimization Done Right | Objective | RL alignment / T5 | Policy weights (no value network) | Remove the response-length and question-difficulty normalization biases in GRPO that artificially inflate response length, giving an unbiased optimization that improves token efficiency while holding reasoning performance ([Liu et al., Understanding R1-Zero](https://arxiv.org/abs/2503.20783)). | Reasoning RL where GRPO's length inflation wastes tokens or degrades efficiency. | High / Medium-High | Native: `trl.GRPOConfig(loss_type="dr_grpo", scale_rewards=False)`; the TRL docs cite the Dr. GRPO paper for the no-reward-scaling recommendation ([TRL GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer)). | A targeted bias fix on GRPO rather than a standalone algorithm; new (March 2025). | emerging |

### Preference/RL, distillation, and efficient full-parameter methods (ranks 18 to 34)

| # | Method (acronym) | Type | Family / Tier | Parameter scope | Core mechanism (one sentence) | Best use cases | Compute / VRAM | Python implementation mapping | Principal limitation | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| 18 | Domain-/Task-Adaptive Pretraining (DAPT/TAPT) | Objective | Continued pretraining / T5-T6 | All weights | Add a second in-domain (DAPT) and/or task-corpus (TAPT) LM-pretraining phase before fine-tuning for consistent task gains ([Gururangan et al., DAPT/TAPT](https://arxiv.org/abs/2004.10964)). | Specializing a base model to a domain (bio, legal, code) cheaply. | High / Medium-High | Custom or Native: HF `Trainer` MLM or CLM objective on the domain corpus ([Gururangan et al., DAPT/TAPT](https://arxiv.org/abs/2004.10964)). | Needs a curated domain corpus; forgetting risk. | established |
| 19 | Knowledge Distillation (KD, soft targets) | Objective | Distillation / T5 | Student weights | Train a small student to match a large teacher's softened output distribution, compressing teacher knowledge into a deployable model ([Hinton et al., KD](https://arxiv.org/abs/1503.02531)). | Model compression; training small models from a strong teacher. | Medium / Medium | Custom: KL(student || teacher) on logits plus CE; teacher forward pass in the loop ([Hinton et al., KD](https://arxiv.org/abs/1503.02531)). | Needs teacher access; capacity gap limits fidelity. | established |
| 20 | IPO, Identity Preference Optimization | Objective | Preference optimization / T5 | Policy weights (+ reference) | A general preference objective (Psi-PO) expressed directly in pairwise preferences that, with the identity mapping, avoids DPO's overfitting to near-deterministic preferences ([Azar et al., IPO](https://arxiv.org/abs/2310.12036)). | Robust offline alignment where preferences are strong or noisy. | Medium / Medium | Native: `trl.DPOTrainer(loss_type="ipo")` ([TRL index](https://huggingface.co/docs/trl/index); [Azar et al., IPO](https://arxiv.org/abs/2310.12036)). | Still an offline pairwise method; extra regularization tuning. | established |
| 21 | RLOO, REINFORCE Leave-One-Out | Objective | RL alignment / T5 | Policy weights (no value network) | Replace PPO's critic and heavy machinery with simpler REINFORCE-style optimization using a leave-one-out baseline over sampled completions ([Ahmadian et al., RLOO](https://arxiv.org/abs/2402.14740)). | Cheaper online RLHF than PPO with equal or better quality. | High / Medium | Native: `trl.RLOOTrainer` ([TRL index](https://huggingface.co/docs/trl/index)). | Still requires online sampling and a reward model. | emerging |
| 22 | REINFORCE++ | Objective | RL alignment / T5 | Policy weights (no value network, no critic) | A critic-free policy-optimization framework centered on Global Advantage Normalization: normalize advantages across the entire global batch rather than small prompt-specific groups for a more stable, effectively unbiased estimate ([Hu et al., REINFORCE++](https://arxiv.org/abs/2501.03262)). | General-domain RLHF and agentic settings where per-group normalization (GRPO/RLOO) is biased or unstable. | High / Medium | Custom: implement global-batch advantage normalization in a REINFORCE loop; reference implementation in the OpenRLHF ecosystem per the paper ([Hu et al., REINFORCE++](https://arxiv.org/abs/2501.03262)). | Not a first-class TRL trainer; the global-normalization benefit grows with batch size. | emerging |
| 23 | Prompt Tuning (soft prompts) | Parameterization | Soft-prompt / T1 | Frozen base + a learned soft-prompt embedding | Learn continuous soft-prompt vectors by backprop through a frozen LM; competitiveness with full tuning grows with model scale ([Lester et al., Prompt Tuning](https://arxiv.org/abs/2104.08691)). | Cheap per-task conditioning on very large frozen models. | Very low / Very low | Native: `peft.PromptTuningConfig` ([PEFT PEFT types](https://huggingface.co/docs/peft/v0.17.0/en/package_reference/peft_types)). | Weak for small models or hard tasks ([Lester et al., Prompt Tuning](https://arxiv.org/abs/2104.08691)). | established |
| 24 | Prefix-Tuning | Parameterization | Soft-prompt / T1 | Frozen base + continuous prefix vectors at every layer | Prepend trainable continuous virtual-token vectors that all layers attend to, training only about 0.1% of parameters while keeping the LM frozen ([Li & Liang, Prefix-Tuning](https://arxiv.org/abs/2101.00190)). | Multi-task serving from one frozen model; generation tasks. | Very low / Very low | Native: `peft.PrefixTuningConfig` ([PEFT PEFT types](https://huggingface.co/docs/peft/v0.17.0/en/package_reference/peft_types)). | Weaker on hard tasks; consumes context length. | established |
| 25 | P-Tuning v2 (deep prompt tuning) | Parameterization | Soft-prompt / T1 | Frozen base + prompts at every layer | Apply continuous prompts at all layers (deep prompt tuning) so prompt tuning matches full FT across scales and tasks with only 0.1 to 3% of parameters ([Liu et al., P-Tuning v2](https://arxiv.org/abs/2110.07602)). | NLU and sequence labeling with a frozen backbone. | Very low / Low | Native: implemented via PEFT prefix-style deep prompts; PEFT also exposes the original `P_TUNING` prompt-encoder type ([PEFT PEFT types](https://huggingface.co/docs/peft/v0.17.0/en/package_reference/peft_types)). | More parameters than shallow prompt tuning. | established |
| 26 | LoftQ | Recipe | Quantized low-rank init / T4 | Frozen quantized base + LoRA (quant-aware init) | Jointly quantize the LLM and choose a LoRA initialization that closes the quantized-to-full-precision gap, excelling in 2-bit and mixed regimes ([Li et al., LoftQ](https://arxiv.org/abs/2310.08659)). | Aggressive (4-bit or lower) QLoRA where the quantization gap hurts. | Low / Very low | Native: PEFT LoftQ init utilities for `LoraConfig` ([Li et al., LoftQ](https://arxiv.org/abs/2310.08659)). | Init cost; still a low-rank subspace. | emerging |
| 27 | PiSSA, Principal Singular Values and Vectors Adaptation | Parameterization | Low-rank PEFT / T2 (+T4 as QPiSSA) | Frozen residual + LoRA init on principal components | Initialize the LoRA adapters from the principal singular components of W (freezing the residual), giving faster convergence and higher accuracy than noise-and-zero LoRA ([Meng et al., PiSSA](https://arxiv.org/abs/2404.02948)). | LoRA replacement when convergence or quality matters; QPiSSA for quantized. | Low / Low | Native: PEFT PiSSA initialization for `LoraConfig` (`init_lora_weights="pissa"`) ([Meng et al., PiSSA](https://arxiv.org/abs/2404.02948)). | Requires an SVD of the base (fast but non-trivial); low-rank limit remains. | emerging |
| 28 | Adapters (Houlsby bottleneck) | Parameterization | Adapter PEFT / T2 | Frozen base + inserted bottleneck adapter modules | Insert small trainable bottleneck modules between layers, adding only a few percent of parameters per task with high parameter sharing across tasks ([Houlsby et al., Adapters](https://arxiv.org/abs/1902.00751)). | Multi-task adaptation; the original PEFT paradigm. | Low / Low | Native: `adapters` library or a custom `nn.Module` bottleneck ([Houlsby et al., Adapters](https://arxiv.org/abs/1902.00751)). | Adds inference latency, unlike LoRA ([Hu et al., LoRA](https://arxiv.org/abs/2106.09685)). | established |
| 29 | IA3, Infused Adapter by Inhibiting and Amplifying Activations | Parameterization | Adapter PEFT / T2 | Frozen base + learned activation-scaling vectors | Scale keys, values, and FFN activations by learned vectors, adding a tiny parameter count while beating few-shot in-context learning (the T-Few recipe) ([Liu et al., IA3](https://arxiv.org/abs/2205.05638)). | Ultra-low-parameter few-shot adaptation. | Very low / Very low | Native: `peft.IA3Config` ([PEFT PEFT types](https://huggingface.co/docs/peft/v0.17.0/en/package_reference/peft_types)). | Limited capacity for large behavioral shifts. | established |
| 30 | RLAIF / Constitutional AI | Pipeline | RL alignment (AI feedback) / T5 | Policy (+ AI preference model) | Replace human harmfulness labels with a rule-guided AI that generates critiques and revisions and preference labels, then RL against that AI preference model ([Bai et al., Constitutional AI](https://arxiv.org/abs/2212.08073)). | Scalable harmlessness and alignment with minimal human labels. | Extreme / High | Custom pipeline: self-critique and revision SFT, then a TRL reward or GRPO/PPO stage on AI-labeled preferences ([Bai et al., Constitutional AI](https://arxiv.org/abs/2212.08073)). | Inherits the labeling model's biases; pipeline complexity. | established |
| 31 | RAFT, Reward-rAnked Fine-Tuning (rejection sampling) | Pipeline | Self-training / preference / T5 | All weights (SFT on filtered samples) | Sample many completions, keep only the reward-model-selected high-quality ones, and SFT on them, a stable RL-free alignment loop ([Dong et al., RAFT](https://arxiv.org/abs/2304.06767)). | Simple alignment or self-improvement without RL instability. | Medium / Medium | Custom: generate, then reward-filter, then `trl.SFTTrainer`, then iterate ([Dong et al., RAFT](https://arxiv.org/abs/2304.06767)). | Needs a reward model and heavy sampling; can narrow diversity. | established |
| 32 | STaR, Self-Taught Reasoner | Pipeline | Self-training / T5 | All weights (SFT on correct rationales) | Bootstrap reasoning by generating rationales, rationalizing from correct answers when wrong, and fine-tuning on rationales that yield correct answers, iterating ([Zelikman et al., STaR](https://arxiv.org/abs/2203.14465)). | Improving reasoning with few rationale seeds plus answer-checkable data. | Medium / Medium | Custom: generate CoT, filter by answer correctness, then SFT, in a loop ([Zelikman et al., STaR](https://arxiv.org/abs/2203.14465)). | Needs verifiable answers; can reinforce spurious rationales. | established |
| 33 | Self-Instruct | Pipeline | Synthetic-data self-training / T5 | All weights (SFT on self-generated instructions) | Bootstrap an instruction dataset from the model's own generations, filter it, and fine-tune the same model, an almost annotation-free instruction-tuning method ([Wang et al., Self-Instruct](https://arxiv.org/abs/2212.10560)). | Cheaply creating instruction data to SFT a base model. | Medium / Medium | Custom: generate and filter instructions, then `trl.SFTTrainer` ([Wang et al., Self-Instruct](https://arxiv.org/abs/2212.10560)). | Quality and diversity capped by the generator; error amplification. | established |
| 34 | GKD, Generalized Knowledge Distillation (on-policy) | Objective | Distillation / T5 | Student weights | Train the student on its own generated sequences with teacher feedback (fixing the train-inference distribution mismatch) and flexible divergences, integrable with RLHF ([Agarwal et al., GKD](https://arxiv.org/abs/2306.13649)). | On-policy distillation of autoregressive LLMs. | High / Medium-High | Native: `trl.GKDTrainer` (marked experimental in current docs) ([TRL index](https://huggingface.co/docs/trl/index)). | Student sampling in the loop is costly; needs a white-box teacher. | emerging |

### Efficient full-parameter, continual-learning, specialized PEFT, and pretraining (ranks 35 to 50)

| # | Method (acronym) | Type | Family / Tier | Parameter scope | Core mechanism (one sentence) | Best use cases | Compute / VRAM | Python implementation mapping | Principal limitation | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| 35 | MiniLLM (reverse-KLD distillation) | Objective | Distillation / T5 | Student weights | Distill a white-box LLM by minimizing reverse KLD via on-policy optimization so the student stops overestimating the teacher's low-probability regions ([Gu et al., MiniLLM](https://arxiv.org/abs/2306.08543)). | White-box LLM to small-LM distillation for generation quality. | High / Medium-High | Native: `trl.MiniLLMTrainer` (marked experimental in current docs) ([TRL index](https://huggingface.co/docs/trl/index)). | On-policy cost; white-box teacher required. | emerging |
| 36 | LoRA+ | Parameterization | Low-rank PEFT / T2 | Frozen base + LoRA with different LRs for A and B | Set a well-chosen ratio of learning rates for the A versus B LoRA matrices to enable efficient feature learning in wide models (1 to 2% gain, up to about 2x speedup, same cost) ([Hayou et al., LoRA+](https://arxiv.org/abs/2402.12354)). | Drop-in LoRA tweak for large-width models. | Low / Low | Custom: per-parameter-group LR in the optimizer over LoRA A and B ([Hayou et al., LoRA+](https://arxiv.org/abs/2402.12354)). | Requires tuning the LR ratio. | emerging |
| 37 | rsLoRA, Rank-Stabilized LoRA | Parameterization | Low-rank PEFT / T2 | Frozen base + LoRA with sqrt(r) scaling | Divide the LoRA update by sqrt(r) instead of r so higher ranks train stably, unlocking a compute-quality trade-off at fixed inference cost ([Kalajdzievski, rsLoRA](https://arxiv.org/abs/2312.03732)). | Using higher-rank LoRA for more capacity. | Low-Medium / Low | Native: `peft.LoraConfig(use_rslora=True)` ([PEFT LoRA reference](https://huggingface.co/docs/peft/package_reference/lora)). | Higher rank costs more compute and memory. | established |
| 38 | BitFit | Parameterization | Sparse PEFT / T3 | Only bias terms | Fine-tune only the model's bias terms, competitive with full FT on small and medium data ([Ben-Zaken et al., BitFit](https://arxiv.org/abs/2106.10199)). | Extremely lightweight adaptation; probing what fine-tuning changes. | Very low / Very low | Custom: set `requires_grad=False` except on `*.bias` ([Ben-Zaken et al., BitFit](https://arxiv.org/abs/2106.10199)). | Limited capacity; weaker on large data ([Ben-Zaken et al., BitFit](https://arxiv.org/abs/2106.10199)). | established |
| 39 | LISA, Layerwise Importance Sampled AdamW | Full-parameter training strategy | Memory-efficient full-parameter / T5 | A randomly sampled subset of layers per step | Randomly freeze most middle layers each step (importance sampling over layers) so memory is LoRA-like while training reaches or exceeds full FT ([Pan et al., LISA](https://arxiv.org/abs/2403.17919)). | Full-quality FT at LoRA-level memory; no merge step. | Medium / Low | Custom: per-step activate and freeze layer groups plus AdamW ([Pan et al., LISA](https://arxiv.org/abs/2403.17919)). | Sampling-schedule sensitivity; less mature tooling. | emerging |
| 40 | LOMO, Low-Memory Optimization | Full-parameter training strategy | Memory-efficient full-parameter / T5 | All weights (fused SGD) | Fuse gradient computation and parameter update into one step to slash optimizer memory, enabling full-parameter FT of a 65B model on 8x24GB GPUs ([Lv et al., LOMO](https://arxiv.org/abs/2306.09782)). | Full FT of very large models on modest multi-GPU boxes. | High / Low | Custom: the LOMO fused-update optimizer (OpenLMLab reference repo) ([Lv et al., LOMO](https://arxiv.org/abs/2306.09782)). | SGD-like dynamics (no Adam states) can slow convergence. | emerging |
| 41 | MeZO, Memory-efficient Zeroth-Order Optimizer | Full-parameter training strategy | Memory-efficient full-parameter / T5 | All weights (or +PEFT) via forward-only | Estimate gradients from two forward passes (in-place ZO-SGD) so fine-tuning uses inference-level memory, training a 30B model where backprop fits only 2.7B ([Malladi et al., MeZO](https://arxiv.org/abs/2305.17333)). | Extreme-memory-constrained FT; non-differentiable objectives. | Medium (many steps) / Very low | Custom: the MeZO zeroth-order perturbation loop (reference implementation) ([Malladi et al., MeZO](https://arxiv.org/abs/2305.17333)). | Slow convergence; noisy ZO estimates. | emerging |
| 42 | VeRA, Vector-based Random Matrix Adaptation | Parameterization | Low-rank PEFT / T2 | Frozen shared random matrices + trainable scaling vectors | Share a single frozen random low-rank matrix pair across all layers and train only small scaling vectors, drastically cutting stored parameters versus LoRA at similar quality ([Kopiczko et al., VeRA](https://arxiv.org/abs/2310.11454)). | Massive multi-adapter or per-user deployment where storage dominates. | Low / Very low | Native: `peft.VeraConfig` ([PEFT PEFT types](https://huggingface.co/docs/peft/v0.17.0/en/package_reference/peft_types)). | Capacity limited by scaling-vector-only training. | emerging |
| 43 | LoReFT, Low-rank Linear Subspace Representation Finetuning | Parameterization | Representation finetuning / T1 | Frozen base + learned interventions on hidden states | Learn low-rank linear interventions on frozen hidden representations rather than weights, reportedly 15 to 65x more parameter-efficient than LoRA ([Wu et al., ReFT](https://arxiv.org/abs/2404.03592)). | Ultra-parameter-efficient adaptation; interpretability-aligned edits. | Very low / Very low | Native: the `pyreft` library ([Wu et al., ReFT](https://arxiv.org/abs/2404.03592)). | Newer; less battle-tested tooling and serving. | research-stage |
| 44 | BOFT / OFT, Orthogonal (Butterfly) Finetuning | Parameterization | Structured PEFT / T2 | Frozen base + orthogonal transform of weights (butterfly-factorized) | Adapt weights via multiplicative orthogonal transforms, made parameter-efficient with butterfly factorization (BOFT subsumes OFT) ([Liu et al., BOFT](https://arxiv.org/abs/2311.06243)). | Adaptation preserving pretrained geometry (also strong for diffusion). | Low / Low | Native: `peft.BOFTConfig` / `OFTConfig` ([PEFT PEFT types](https://huggingface.co/docs/peft/v0.17.0/en/package_reference/peft_types)). | More complex than LoRA; niche for text LLMs. | emerging |
| 45 | MiSS, Matrix Shard Sharing | Parameterization | PEFT / T2 | Frozen base + a single trainable shard-sharing matrix (mergeable) | Decompose a weight matrix into fragments and learn a shared trainable common fragment, reconstructing a high-capacity update from replicated shards; an evolution of Bone that PEFT now recommends in its place ([PEFT release notes, MiSS](https://github.com/huggingface/peft/releases); [MiSS paper](https://huggingface.co/papers/2409.15371)). | LoRA-alternative PEFT prioritizing performance-to-memory efficiency; the current recommended successor to Bone. | Low / Low | Native: `peft.MissConfig` (Bone is deprecated and slated for removal in favor of MiSS) ([PEFT release notes, MiSS](https://github.com/huggingface/peft/releases); [PEFT PEFT types](https://huggingface.co/docs/peft/v0.17.0/en/package_reference/peft_types)). | Very new (2025 addition); smaller track record than LoRA. | emerging |
| 46 | SPIN, Self-Play Fine-Tuning | Pipeline | Self-training / T5 | Policy weights | Iterative self-play in which the model discriminates its previous-iteration self-generations from human SFT data, converging to the target distribution without new human labels ([Chen et al., SPIN](https://arxiv.org/abs/2401.01335)). | Squeezing more from a fixed SFT set; label-free improvement. | High / Medium | Custom: a DPO-style loss with self-generated loser responses, iterated ([Chen et al., SPIN](https://arxiv.org/abs/2401.01335)). | Iterative sampling cost; can plateau or collapse. | emerging |
| 47 | EWC, Elastic Weight Consolidation | Modifier | Continual learning / T5 | All weights, with per-weight regularization | Prevent catastrophic forgetting by adding a Fisher-importance-weighted quadratic penalty that slows changes to weights important for prior tasks ([Kirkpatrick et al., EWC](https://arxiv.org/abs/1612.00796)). | Sequential task or domain learning without storing old data. | Medium / Medium (stores Fisher) | Custom: add the sum of F_i (theta_i - theta*_i)^2 penalty to the loss ([Kirkpatrick et al., EWC](https://arxiv.org/abs/1612.00796)). | Fisher estimation cost; penalty tuning; scales poorly to many tasks. | established |
| 48 | Experience Replay / Rehearsal | Modifier | Continual learning / T5 | All weights, with replayed old data | Interleave a buffer of past examples with new-task training to substantially reduce catastrophic forgetting ([Rolnick et al., Experience Replay](https://arxiv.org/abs/1811.11682)). | Continual or continued training where some old data is retainable. | Medium / Medium (buffer) | Custom: mix replay-buffer batches into the training stream ([Rolnick et al., Experience Replay](https://arxiv.org/abs/1811.11682)). | Requires storing and sampling old data; buffer-size trade-offs. | established |
| 49 | LwF, Learning without Forgetting | Modifier | Continual learning / distillation / T5 | All weights | Use only new-task data plus distillation from the old model's outputs to preserve prior capabilities without old training data ([Li & Hoiem, LwF](https://arxiv.org/abs/1606.09282)). | Adding capabilities when old data is unavailable. | Medium / Medium | Custom: a KD loss against frozen old-model outputs on new data ([Li & Hoiem, LwF](https://arxiv.org/abs/1606.09282)). | Distillation targets drift under large domain shift. | established |
| 50 | Causal-LM Pretraining from Scratch (compute-optimal) | Objective | From-scratch pretraining / T6 | All weights, trained from random initialization | Train all weights from random initialization on a massive general corpus with a next-token (causal-LM) objective, sizing model parameters and training tokens roughly equally per the compute-optimal law ([Hoffmann et al., Chinchilla](https://arxiv.org/abs/2203.15556)). | Building a new base model when no suitable pretrained checkpoint exists and large compute is available. | Extreme / Extreme | Custom or Native: HF `Trainer` (or JAX) with a causal-LM loss over packed raw text at cluster scale; token budget sized by the Chinchilla law ([Hoffmann et al., Chinchilla](https://arxiv.org/abs/2203.15556)). | The most expensive option by orders of magnitude; rarely justified over continued pretraining. | established |

Honorable mentions researched but placed outside the top 50 (documented for transparency): CISPO, which clips importance-sampling weights rather than token updates and is exposed in TRL as `GRPOConfig(loss_type="cispo")` ([MiniMax, MiniMax-M1](https://arxiv.org/abs/2506.13585); [TRL GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer)); BCO, binary-classifier optimization, which overlaps KTO and is available as the experimental `trl.BCOTrainer` ([Jung et al., BCO](https://arxiv.org/abs/2404.04656); [TRL index](https://huggingface.co/docs/trl/index)); NEFTune, a train-time embedding-noise modifier on top of SFT, available as `SFTConfig(neftune_noise_alpha=...)` ([Jain et al., NEFTune](https://arxiv.org/abs/2310.05914); [TRL SFT Trainer](https://huggingface.co/docs/trl/sft_trainer)); RandLoRA, SHiRA, RoAd, and C3A, all 2025 additions in PEFT that are near-neighbors of the ranked low-rank and structured families ([PEFT PEFT types](https://huggingface.co/docs/peft/v0.17.0/en/package_reference/peft_types)); FourierFT (spectral adapters) and MoRA (high-rank square-matrix PEFT), close to the ranked low-rank family; P-Tuning v1 and LLaMA-Adapter, superseded by P-Tuning v2 and by mainstream adapters respectively; Sophia and Adam-mini, which are optimizers, not adaptation strategies ([Liu et al., Sophia](https://arxiv.org/abs/2305.14342); [Zhang et al., Adam-mini](https://arxiv.org/abs/2406.16793)); and the UL2 Mixture-of-Denoisers pretraining objective ([Tay et al., UL2](https://arxiv.org/abs/2205.05131)). The distinction that keeps Sophia and Adam-mini out but keeps GaLore, LOMO, MeZO, and LISA in is explained in the exclusions section.

---

## 2025 to 2026 Changes Since the Prior Draft

This revision reflects developments and documentation states verified through July 21, 2026.

- GRPO now has a family of successors, several selectable inside one TRL trainer. The current `GRPOConfig.loss_type` accepts `grpo`, `dr_grpo`, `dapo` (the current default), `bnpo`, and `cispo`, and `importance_sampling_level` accepts `token` or `sequence` (the latter being GSPO); `delta` enables two-sided Clip-Higher clipping and `scale_rewards=False` follows the Dr. GRPO recommendation ([TRL GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer)). Newly ranked as distinct objectives: DAPO ([Yu et al.](https://arxiv.org/abs/2503.14476)), GSPO ([Zheng et al.](https://arxiv.org/abs/2507.18071)), and Dr. GRPO ([Liu et al.](https://arxiv.org/abs/2503.20783)); REINFORCE++ ([Hu et al.](https://arxiv.org/abs/2501.03262)) is ranked, and CISPO ([MiniMax](https://arxiv.org/abs/2506.13585)) is an honorable mention.
- TRL trainer status changed. In current docs, `PPOTrainer`, `OnlineDPOTrainer`, `NashMDTrainer`, `XPOTrainer`, `PRMTrainer`, `BCOTrainer`, `CPOTrainer`, `ORPOTrainer`, `GKDTrainer`, and `MiniLLMTrainer` are all marked experimental, while `GRPOTrainer`, `RLOOTrainer`, `SFTTrainer`, `DPOTrainer`, `KTOTrainer`, and `RewardTrainer` are not; the docs explicitly note "KTO is now stable" ([TRL index](https://huggingface.co/docs/trl/index)). Prior claims that treated these as stable are corrected in the tables.
- SimPO's mapping is corrected. SimPO is not a standalone TRL trainer; its loss is exposed through the experimental `CPOTrainer` (`loss_type="simpo"`), which the tables now state explicitly ([TRL index](https://huggingface.co/docs/trl/index)).
- PEFT added a 2025 wave of methods. The current supported types include MiSS (which deprecates and replaces Bone), RandLoRA, SHiRA, RoAd, and C3A, alongside the established LoRA, AdaLoRA, BOFT, OFT, IA3, LoHa, LoKr, VeRA, FourierFT, HRA, X-LoRA, Poly, LN Tuning, and the prompt-based methods, plus a standalone Trainable Tokens tuner ([PEFT PEFT types](https://huggingface.co/docs/peft/v0.17.0/en/package_reference/peft_types); [PEFT Trainable Tokens](https://huggingface.co/docs/peft/package_reference/trainable_tokens); [PEFT release notes](https://github.com/huggingface/peft/releases)). MiSS is now ranked; Bone is intentionally not, because it is deprecated in its favor.
- Structural fixes applied per audit. Causal-LM pretraining from scratch is now an explicit ranked entry (rank 50) rather than only a taxonomy note; every entry carries a Type field; the emoji heading and italic styling were removed; and the optimizer-versus-strategy inconsistency is resolved in the exclusions section.

---

## Decision Tree: Choosing a Method

```
START: What is your goal?
|
+- Build/inject broad capability or new domain knowledge at corpus scale?
|    +- From scratch, have a large compute budget?
|    |     -> Causal-LM pretraining, size tokens vs params equally (Chinchilla,
|    |        arxiv 2203.15556).
|    +- Have a strong base model already?
|          -> Continued/Domain-Adaptive Pretraining (DeepSeekMath 2402.03300;
|             DAPT/TAPT 2004.10964). Guard against forgetting: Experience Replay
|             (1811.11682), EWC (1612.00796), or LwF (1606.09282).
|
+- Teach a task/format/instruction-following (you have labeled completions)?
|    +- Need mergeable single-model weights AND max quality, hardware allows?
|    |     -> Full SFT (trl.SFTTrainer). Add NEFTune (2310.05914) nearly for free.
|    |        Memory-tight full-parameter? GaLore (2403.03507),
|    |        LISA (2403.17919), LOMO (2306.09782), or MeZO (2305.17333).
|    +- Memory-constrained / many adapters / must keep base pristine?
|          -> PEFT (below).
|
+- Which PEFT? (base frozen)
|    +- Want the safe, ubiquitous default, mergeable, no latency? -> LoRA (2106.09685)
|    |     +- Close the gap to full FT? -> DoRA (2402.09353) or PiSSA (2404.02948)
|    |     +- Wide model / faster convergence? -> LoRA+ (2402.12354)
|    |     +- Want higher rank stably? -> rsLoRA (2312.03732)
|    |     +- Performance-to-memory-focused LoRA alternative? -> MiSS (peft MissConfig)
|    |     +- Adapter-storage explosion (per-user)? -> VeRA (2310.11454)
|    +- Single 24-48GB GPU + large model? -> QLoRA (2305.14314); aggressive
|    |     4-bit or lower -> LoftQ (2310.08659)
|    +- Ultra-few parameters / few-shot? -> IA3 (2205.05638), BitFit (2106.10199),
|    |     or LoReFT (2404.03592)
|    +- Frozen backbone, multi-task serving from prompts? -> Prefix-Tuning
|          (2101.00190) / Prompt Tuning (2104.08691) / P-Tuning v2 (2110.07602)
|
+- Align to preferences/values (post-SFT)?
     +- Do you have PAIRED preference data?
     |     +- Simplest, offline, robust default? -> DPO (2305.18290);
     |     |     overfitting worry -> IPO (2310.12036)
     |     +- Reference-free / lower memory? -> SimPO (via CPOTrainer, 2405.14734)
     |     |     or ORPO (2403.07691, folds into SFT, one stage)
     |     +- Willing to run online RL for top control/reasoning?
     |           -> GRPO (2402.03300); for long-CoT stability use DAPO (2503.14476)
     |              or GSPO (2507.18071); to remove length bias use Dr. GRPO
     |              (2503.20783); global-normalized critic-free -> REINFORCE++
     |              (2501.03262); cheaper -> RLOO (2402.14740); classic PPO-RLHF
     |              (2203.02155 / 1707.06347)
     +- Only UNPAIRED binary feedback? -> KTO (2402.01306) [or BCO 2404.04656]
     +- Have a reward model but want RL-free? -> RAFT rejection sampling (2304.06767)
     +- Scarce human labels? -> RLAIF / Constitutional AI (2212.08073)
     +- No new labels, verifiable answers or a fixed SFT set?
           -> STaR (2203.14465), Self-Instruct (2212.10560), or SPIN (2401.01335)

DISTILLATION (want a smaller model): white-box teacher -> MiniLLM reverse-KLD
(2306.08543) or on-policy GKD (2306.13649); classic soft-target KD (1503.02531).
```

---

## Terminology: Eight Training Paradigms Compared

All definitions below are grounded in the fetched primary sources.

- Pretraining (from scratch): Train all weights on a massive general corpus with a self-supervised objective (typically causal LM). Compute-optimal practice scales model size and training tokens equally, so doubling parameters should double tokens ([Hoffmann et al., Chinchilla](https://arxiv.org/abs/2203.15556)). The objective can be generalized, for example UL2's Mixture-of-Denoisers ([Tay et al., UL2](https://arxiv.org/abs/2205.05131)).
- Continued pretraining (CPT): Resume the same self-supervised objective on new or large data to add knowledge, for example continuing a base with 120B math tokens ([Shao et al., DeepSeekMath](https://arxiv.org/abs/2402.03300)); domain and task-adaptive variants (DAPT/TAPT) give consistent downstream gains ([Gururangan et al., DAPT/TAPT](https://arxiv.org/abs/2004.10964)). It differs from pretraining only in that it starts from trained weights.
- Supervised fine-tuning (SFT): Supervised learning on demonstration data; the first stage of InstructGPT collected labeler demonstrations and fine-tuned GPT-3 with supervised learning ([Ouyang et al., InstructGPT](https://arxiv.org/abs/2203.02155)); in practice the loss is computed on completion or assistant tokens ([TRL SFT Trainer](https://huggingface.co/docs/trl/sft_trainer)).
- Instruction tuning: A form of SFT where the many training tasks are phrased as natural-language instructions to induce zero-shot generalization to unseen task types ([Wei et al., FLAN](https://arxiv.org/abs/2109.01652)). All instruction tuning is SFT; not all SFT is instruction tuning.
- Preference optimization: Directly optimize the policy from preference or feedback data without an RL rollout loop; DPO recasts RLHF as a classification loss ([Rafailov et al., DPO](https://arxiv.org/abs/2305.18290)); variants use binary feedback (KTO ([Ethayarajh et al.](https://arxiv.org/abs/2402.01306))), reference-free rewards (SimPO ([Meng et al.](https://arxiv.org/abs/2405.14734)), ORPO ([Hong et al.](https://arxiv.org/abs/2403.07691))), or a general pairwise objective (IPO ([Azar et al.](https://arxiv.org/abs/2310.12036))).
- Reinforcement learning (RLHF/RLAIF): Fit a reward model, then optimize the policy with RL to maximize reward without drifting from the reference ([Ouyang et al., InstructGPT](https://arxiv.org/abs/2203.02155)); PPO supplies the surrogate objective ([Schulman et al., PPO](https://arxiv.org/abs/1707.06347)), while GRPO ([Shao et al.](https://arxiv.org/abs/2402.03300)) and its 2025 successors DAPO, GSPO, Dr. GRPO, REINFORCE++, and CISPO drop the value network and refine the advantage or importance estimator ([Yu et al.](https://arxiv.org/abs/2503.14476); [Zheng et al.](https://arxiv.org/abs/2507.18071); [Liu et al.](https://arxiv.org/abs/2503.20783); [Hu et al.](https://arxiv.org/abs/2501.03262); [MiniMax](https://arxiv.org/abs/2506.13585)), and RLAIF replaces human labels with AI feedback ([Bai et al.](https://arxiv.org/abs/2212.08073)). The key contrast with preference optimization is online sampling plus reward maximization versus an offline direct loss.
- Distillation: Train a usually smaller student to match a teacher's output distribution ([Hinton et al., KD](https://arxiv.org/abs/1503.02531)); modern LLM variants use reverse KLD (MiniLLM ([Gu et al.](https://arxiv.org/abs/2306.08543))) or on-policy student generations (GKD ([Agarwal et al.](https://arxiv.org/abs/2306.13649))). Distillation transfers a teacher's behavior rather than learning from labels or preferences directly.
- Parameter-efficient fine-tuning (PEFT): Adapt a model by training a small number of extra parameters while freezing most weights, cutting compute and storage with comparable quality ([PEFT PEFT types](https://huggingface.co/docs/peft/v0.17.0/en/package_reference/peft_types)). PEFT is orthogonal to the above: you can do PEFT-SFT, PEFT-DPO, PEFT-CPT, and so on. It defines which parameters change, whereas the others define what objective is optimized.

---

## Recommended Default Stacks

Hardware-specific defaults. MLX claims are limited to what its documentation verifies.

(a) Single 24 to 48 GB GPU (for example RTX 4090 or A6000).
- Base fine-tune: QLoRA, a 4-bit NF4 base plus LoRA, explicitly designed to fit large models on a 48GB GPU ([Dettmers et al., QLoRA](https://arxiv.org/abs/2305.14314)), via `bitsandbytes` + `peft.LoraConfig` ([PEFT LoRA reference](https://huggingface.co/docs/peft/package_reference/lora)). Add DoRA (`use_dora=True`) or PiSSA init to close the quality gap ([Liu et al., DoRA](https://arxiv.org/abs/2402.09353); [Meng et al., PiSSA](https://arxiv.org/abs/2404.02948)).
- Want full-parameter on 24GB? GaLore enables 7B pretraining on a 24GB RTX 4090 ([Zhao et al., GaLore](https://arxiv.org/abs/2403.03507)); LISA matches full FT at LoRA memory ([Pan et al., LISA](https://arxiv.org/abs/2403.17919)).
- Alignment: DPO ([Rafailov et al., DPO](https://arxiv.org/abs/2305.18290)) or reference-free ORPO/SimPO to save memory ([Hong et al., ORPO](https://arxiv.org/abs/2403.07691); [Meng et al., SimPO](https://arxiv.org/abs/2405.14734)), noting SimPO runs through the experimental `CPOTrainer` ([TRL index](https://huggingface.co/docs/trl/index)). Add NEFTune during SFT ([Jain et al., NEFTune](https://arxiv.org/abs/2310.05914)).

(b) Multi-GPU workstation or server (for example 2 to 8 A100/H100).
- Base: Full SFT with `trl.SFTTrainer` plus packing and assistant-masking ([TRL SFT Trainer](https://huggingface.co/docs/trl/sft_trainer)), distributed via FSDP or DeepSpeed ZeRO (infrastructure; see the warning). For 65B on 8x24GB, LOMO ([Lv et al., LOMO](https://arxiv.org/abs/2306.09782)).
- Alignment: DPO for offline pairs, or online GRPO for reasoning and verifiable rewards, choosing the loss variant via `GRPOConfig` (DAPO or GSPO for long-CoT and MoE stability, Dr. GRPO to remove length bias) ([Shao et al., GRPO](https://arxiv.org/abs/2402.03300); [TRL GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer)); RLOO or full RLHF-PPO if you need cheaper or maximal control ([Ahmadian et al., RLOO](https://arxiv.org/abs/2402.14740); [Ouyang et al., InstructGPT](https://arxiv.org/abs/2203.02155)).
- Distillation to smaller models: GKD or MiniLLM (both experimental in TRL) ([Agarwal et al., GKD](https://arxiv.org/abs/2306.13649); [Gu et al., MiniLLM](https://arxiv.org/abs/2306.08543); [TRL index](https://huggingface.co/docs/trl/index)).

(c) Apple Silicon local experimentation (MLX).
- Verified in mlx-lm docs: the currently supported fine-tune types are LoRA (default), DoRA, and full, and training uses QLoRA automatically when the model is quantized. These are the only fine-tune types the documentation lists ([mlx-lm LORA docs](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)).
- Do not assume MLX supports DPO, ORPO, or GRPO. The mlx-lm LoRA documentation does not mention DPO or ORPO ([mlx-lm LORA docs](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)). For preference optimization or RL on Apple Silicon, implement the loss yourself or move to a PyTorch/TRL setup.
- Default: LoRA or QLoRA fine-tuning for small and medium models; DoRA if you need extra quality; full FT only for small models.

(d) Maximal-quality cluster training.
- Pretraining from scratch: a causal-LM objective sized by the compute-optimal law (tokens proportional to parameters) ([Hoffmann et al., Chinchilla](https://arxiv.org/abs/2203.15556)); consider UL2 Mixture-of-Denoisers for a unified objective ([Tay et al., UL2](https://arxiv.org/abs/2205.05131)).
- Post-training: Full SFT, then full RLHF (PPO) or GRPO-family RL, with RLAIF/Constitutional AI for scalable oversight ([Ouyang et al., InstructGPT](https://arxiv.org/abs/2203.02155); [Shao et al., GRPO](https://arxiv.org/abs/2402.03300); [Bai et al., Constitutional AI](https://arxiv.org/abs/2212.08073)). For MoE models specifically, GSPO stabilizes RL where token-level ratios fail ([Zheng et al., GSPO](https://arxiv.org/abs/2507.18071)). Manage forgetting across stages with Experience Replay or EWC ([Rolnick et al.](https://arxiv.org/abs/1811.11682); [Kirkpatrick et al.](https://arxiv.org/abs/1612.00796)).

---

## Warning: Infrastructure Techniques Are Not Fine-Tuning Methods

The following are enablers. They reduce the cost or increase the scale of any method above but define no training objective or parameterization, and are therefore excluded from the ranking:

- Gradient checkpointing / activation recomputation: trades compute for activation memory.
- Mixed precision (bf16/fp16): a numerical format, not an adaptation strategy.
- ZeRO and FSDP (Fully Sharded Data Parallel): distributed sharding of states, parameters, and gradients; PEFT documents FSDP and DeepSpeed as how-to integrations for scaling, not as PEFT methods ([PEFT main index](https://huggingface.co/docs/peft/main/en/index)).
- Tensor / pipeline parallelism: model partitioning for scale.
- FlashAttention: an exact-attention kernel; it changes speed and memory, not the objective.
- Sequence packing: an efficiency device (`packing=True` in `SFTConfig` groups sequences into fixed blocks to reduce padding) layered on top of SFT ([TRL SFT Trainer](https://huggingface.co/docs/trl/sft_trainer)).
- Optimizer-state offload / paged optimizers: the CPU-offload and paged-optimizer tricks (the latter is one component of the QLoRA recipe ([Dettmers et al., QLoRA](https://arxiv.org/abs/2305.14314))) manage memory spikes; they are not standalone fine-tuning methods.

Two honest edge cases: QLoRA blends an infrastructure technique (4-bit quantization plus paged optimizers) with a parameterization (LoRA), but is established enough to rank as a recipe (rank 3), and the tables say so explicitly ([Dettmers et al., QLoRA](https://arxiv.org/abs/2305.14314)). Sophia and Adam-mini are optimizers; they change the update rule, not what is trained, so they are honorable mentions, not ranked methods ([Liu et al., Sophia](https://arxiv.org/abs/2305.14342); [Zhang et al., Adam-mini](https://arxiv.org/abs/2406.16793)).

---

## Generic Python Training Architecture (where each method plugs in)

The pseudocode below is a single training-script skeleton; the inline comments mark the exact insertion points for each ranked method. It is framework-realistic (PyTorch plus HF Transformers/PEFT/TRL) but abstract enough to map any row into code.

```python
# -- 1. MODEL LOADING --------------------------------------------------------
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# QLoRA / LoftQ: load the base in 4-bit here (bitsandbytes).
#   quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", ...)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16,
                                             # quantization_config=quantization_config
                                             )
tok = AutoTokenizer.from_pretrained(MODEL_ID)
# Causal-LM pretraining from scratch (rank 50): build model from a config instead:
#   model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(ARCH))

# -- 2. PARAMETER SELECTION / INJECTION (defines "what is trained") ----------
# --- PEFT (T1/T2/T3): freeze base, inject or select trainable params ---
from peft import LoraConfig, get_peft_model, TaskType
peft_cfg = LoraConfig(r=16, lora_alpha=32, task_type=TaskType.CAUSAL_LM,
                      use_dora=True,      # -> DoRA (2402.09353)
                      use_rslora=True,    # -> rsLoRA (2312.03732)
                      # init_lora_weights="pissa" or "loftq" -> PiSSA / LoftQ
                      )
model = get_peft_model(model, peft_cfg)   # LoRA/DoRA/AdaLoRA/VeRA/IA3/BOFT/MiSS
# BitFit:  for n,p in model.named_parameters(): p.requires_grad = n.endswith(".bias")
# LISA:    per-step randomly set requires_grad on a sampled subset of layers (2403.17919)
# Prompt/Prefix/P-Tuning v2: PromptTuningConfig / PrefixTuningConfig / PromptEncoderConfig (T1)
# LoReFT:  wrap with pyreft interventions on hidden states (2404.03592)
# Full FT / CPT / Pretraining: skip PEFT -- all params trainable (T5/T6)

# -- 3. DATASET & COLLATOR (defines the data contract for the objective) -----
# SFT/Instruction: prompt-completion or chat; mask prompt (completion_only_loss / assistant_only_loss)
# Preference (DPO/IPO/ORPO/SimPO): {"prompt","chosen","rejected"} triples
# KTO/BCO: {"prompt","completion","label"} unpaired binary
# RL (PPO/GRPO/DAPO/GSPO/Dr.GRPO/RLOO/REINFORCE++/RAFT/RLAIF): prompts + reward fn/model + sampler
# KD/GKD/MiniLLM: same inputs + a frozen TEACHER model in the loop
# CPT/Pretraining: raw text streamed, packed to fixed length

# -- 4. OBJECTIVE / LOSS (the METHOD itself) ---------------------------------
# Prefer a TRL trainer when native support exists (experimental flag noted in docs):
from trl import SFTTrainer, SFTConfig            # SFT/Instruction (+neftune_noise_alpha -> NEFTune)
# from trl import DPOTrainer   # DPO (loss_type="ipo" -> IPO)
# from trl import ORPOTrainer, KTOTrainer, BCOTrainer, CPOTrainer  # ORPO/KTO/BCO/(SimPO via CPO loss_type="simpo")
# from trl import GRPOTrainer, GRPOConfig        # GRPO family; select variant via config:
#     GRPOConfig(loss_type="dapo"|"dr_grpo"|"bnpo"|"cispo",         # DAPO(default)/Dr.GRPO/CISPO
#                importance_sampling_level="sequence",              # -> GSPO (2507.18071)
#                delta=..., scale_rewards=False, mask_truncated_completions=True)
# from trl import RLOOTrainer, PPOTrainer, RewardTrainer            # other online RL
# from trl import GKDTrainer, MiniLLMTrainer      # distillation (experimental)
#
# CUSTOM losses when no native class exists -- implement the equation:
#   KD (1503.02531):    L = a*CE(y) + (1-a)*T^2*KL(softmax(z_s/T) || softmax(z_t/T))
#   EWC (1612.00796):   L = L_task + sum_i (lambda/2)*F_i*(theta_i - theta*_i)^2
#   LwF (1606.09282):   L = L_new + KD(old_model_logits, student_logits)  # new data only
#   Replay (1811.11682): mix batches from a stored buffer of past examples
#   REINFORCE++ (2501.03262): REINFORCE with global-batch advantage normalization
#   STaR/RAFT/SPIN/Self-Instruct/RLAIF: outer generate->filter/label loop, inner SFT/DPO/RL step
train_args = SFTConfig(bf16=True, packing=True, neftune_noise_alpha=5,
                       completion_only_loss=True,
                       gradient_checkpointing=True)   # <- INFRASTRUCTURE flags (enablers)

# -- 5. OPTIMIZER (update rule -- NOT the method by itself) -------------------
# Default: AdamW.  Memory-efficient full-parameter STRATEGIES live here or in step 2:
#   GaLore -> optim="galore_adamw" (2403.03507)   LOMO -> fused SGD update (2306.09782)
#   MeZO -> zeroth-order forward-only loop (2305.17333)
#   (Bare optimizer swaps like Sophia / Adam-mini are enablers, not ranked methods.)

# -- 6. DISTRIBUTED STRATEGY (enabler) ---------------------------------------
#   FSDP / DeepSpeed ZeRO / tensor+pipeline parallelism via accelerate config -- scale only.

# -- 7. TRAIN / EVAL ---------------------------------------------------------
trainer = SFTTrainer(model=model, args=train_args, train_dataset=ds, processing_class=tok)
trainer.train()
# Eval: task metrics / MT-Bench-style / reward, plus a forgetting probe for CPT/continual runs.

# -- 8. CHECKPOINT / EXPORT --------------------------------------------------
model.save_pretrained(OUT)   # PEFT -> tiny adapter (few MB). merge_and_unload() to fold LoRA/DoRA/MiSS
                             # into the base for latency-free serving (2106.09685).
```

Mapping guidance: step 2 implements families 1 to 3 (soft-prompt, adapter, low-rank/quantized) and the parameter-scope column; step 4 implements families 4 to 6 and 8 (supervised, preference/RL, continual, distillation); steps 5 and 6 are enablers plus the memory-efficient full-parameter strategies (family 7); steps 1, 3, 7, and 8 are common to all. From-scratch and continued pretraining (family 9) use the same skeleton with a raw-text packed dataset and no PEFT.

---

## Questionable / Overlapping Labels and Exclusions

- QLoRA is a recipe, not an atomic method. It is 4-bit quantization plus LoRA plus paged optimizers ([Dettmers et al., QLoRA](https://arxiv.org/abs/2305.14314)). It is included (rank 3) because of its overwhelming practical establishment, with the composition stated honestly and tagged Recipe.
- Why GaLore, LOMO, MeZO, and LISA are ranked but Sophia and Adam-mini are not. The brief forbids padding the list with bare optimizers used without a distinct adaptation strategy. Sophia and Adam-mini only change the numeric update rule for the same trainable parameters and the same objective, so they are enablers ([Liu et al., Sophia](https://arxiv.org/abs/2305.14342); [Zhang et al., Adam-mini](https://arxiv.org/abs/2406.16793)). GaLore, LOMO, MeZO, and LISA are tagged as full-parameter training strategies because they change what is representable or storable during the update: GaLore trains in a projected low-rank gradient subspace ([Zhao et al., GaLore](https://arxiv.org/abs/2403.03507)), LOMO fuses gradient and update to eliminate optimizer state ([Lv et al., LOMO](https://arxiv.org/abs/2306.09782)), MeZO replaces backprop with a zeroth-order forward-only estimator ([Malladi et al., MeZO](https://arxiv.org/abs/2305.17333)), and LISA changes which parameters receive gradients each step ([Pan et al., LISA](https://arxiv.org/abs/2403.17919)). That is a parameter-scope or gradient-structure decision, not a drop-in optimizer swap.
- GRPO successors as distinct entries versus config flags. DAPO, GSPO, and Dr. GRPO are ranked as distinct objectives because each defines a different estimator (token-level clip-higher with dynamic sampling; sequence-level importance ratio; unbiased length normalization), even though TRL exposes them through `GRPOConfig` flags rather than separate trainers ([TRL GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer)). CISPO is kept as an honorable mention because it is closest to being a further loss-type option (`loss_type="cispo"`) rather than a distinct family ([MiniMax, MiniMax-M1](https://arxiv.org/abs/2506.13585)).
- SimPO's implementation caveat. SimPO has no dedicated TRL trainer; it is realized as a `loss_type="simpo"` inside the experimental `CPOTrainer`, which the tables state directly rather than implying a native SimPO class ([TRL index](https://huggingface.co/docs/trl/index)).
- TRL experimental status is not hidden. The tables flag every trainer the current TRL docs mark experimental (PPO, ORPO, CPO, BCO, OnlineDPO, XPO, NashMD, GKD, MiniLLM, PRM), so no method is presented as more production-ready than the docs claim ([TRL index](https://huggingface.co/docs/trl/index)).
- Near-alias PEFT variants collapsed. LoHa, LoKr, X-LoRA, Poly, HRA, RandLoRA, SHiRA, RoAd, C3A, FourierFT, and MoRA all exist in current PEFT or the literature but are near-neighbors of ranked low-rank, structured, or adapter families ([PEFT PEFT types](https://huggingface.co/docs/peft/v0.17.0/en/package_reference/peft_types)); the most distinct and established representatives are ranked and the rest are honorable mentions. Bone is deliberately not ranked because PEFT deprecates it in favor of MiSS, which is ranked ([PEFT release notes](https://github.com/huggingface/peft/releases)).
- P-Tuning v1 and LLaMA-Adapter dropped from the ranking. P-Tuning v1 is superseded by P-Tuning v2 on hard tasks ([Liu et al., P-Tuning v2](https://arxiv.org/abs/2110.07602)) and LLaMA-Adapter is architecture-specific and largely displaced by mainstream adapters; both are noted as honorable mentions.
- PPO versus RLHF. PPO ([Schulman et al., PPO](https://arxiv.org/abs/1707.06347)) is the optimizer inside RLHF; the RLHF pipeline with PPO is ranked once rather than double-counted.
- Sequence-level KD ([Kim & Rush](https://arxiv.org/abs/1606.07947)) is folded into the KD/GKD/MiniLLM cluster rather than listed separately.
- UL2 Mixture-of-Denoisers ([Tay et al., UL2](https://arxiv.org/abs/2205.05131)) is a distinct pretraining objective but overlaps the causal-LM pretraining entry; it is an honorable-mention alternative objective, kept out to hold exactly 50.
- Infrastructure techniques (checkpointing, mixed precision, ZeRO/FSDP, parallelism, FlashAttention, packing, offload) are excluded by design and treated in the warning section.

---

## Conclusions

The 2026 landscape is best understood as a matrix, not a list: an objective axis (pretraining, continued pretraining, SFT/instruction tuning, preference optimization, RL, distillation) crossed with a parameter-scope axis (full weights, quantized low-rank, additive PEFT, sparse/structural, representation/prompt). Any cell is a valid method; the top methods are those that are simultaneously high-utility, mature, broadly used, efficient, and conceptually distinct. LoRA and QLoRA dominate parameter scope; SFT, DPO, and the GRPO family dominate objectives; GaLore, LISA, LOMO, and MeZO have re-enabled full-parameter work on small hardware; and direct and critic-free preference methods have largely displaced the operational burden of classic PPO-RLHF, while PPO-RLHF and RLAIF remain the ceiling for controllability.

The clearest 2025-2026 signal is convergence on the GRPO family for reasoning RL, now with a menu of well-characterized estimators (DAPO, GSPO, Dr. GRPO, REINFORCE++, CISPO) that a practitioner selects mostly by editing a config rather than swapping frameworks ([TRL GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer)). On the PEFT side, the field keeps generating low-rank and structured variants (MiSS, RandLoRA, SHiRA, RoAd, C3A) faster than any of them displaces LoRA as the default.

The most important discipline for practitioners is the one the brief demands: separate methods from enablers, and separate objectives from parameterizations from recipes from pipelines from modifiers. Choosing DPO versus GRPO, or LoRA versus full FT, is a method decision that changes your model's learned behavior; enabling FSDP, FlashAttention, or gradient checkpointing only changes whether that decision fits in your budget. Get the method right first, then let the infrastructure make it affordable.

Where a specific numeric value could not be confirmed from a fetched primary source, it is omitted rather than guessed. All citations above link to pages fetched during this research session, with a research cutoff of July 21, 2026.
