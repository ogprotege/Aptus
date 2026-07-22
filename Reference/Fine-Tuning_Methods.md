> **Documentation status:** Archived and rejected as factual authority
>
> **Authority:** Historical name intake only. This file contains known factual
> errors and uncited method descriptions.
>
> **Last reviewed:** 2026-07-22
>
> **Next scheduled review:** 2027-07-22, or when its archive disposition changes
>
> Do not use this file to define a method or planner default. Use the
> [method taxonomy](../docs/methodology/method-taxonomy.md) and read the
> [documented corrections](../docs/research/reference-and-to-review-reconciliation.md#referencefine-tuning_methodsmd).

FineTuning Methods


### General Techniques ###

SFT (Standard Fine-Tuning):

	Description: Involves fine-tuning a pre-trained model on a specific task using supervised data.

	Use Case: Widely used for improving performance on tasks like classification, regression, and sequence generation.


FFT (Fast Fine-Tuning):

	Description: A faster version of fine-tuning that aims to achieve good performance with minimal training time and data.

	Use Case: Useful when training resources are limited or when rapid prototyping is needed.


ReFT (Recursive Fine-Tuning):

	Description: Involves iteratively fine-tuning a model on multiple tasks or datasets.

	Use Case: Suitable for multi-task learning and domain adaptation.





### Regularization and Adaptation Techniques ###


DoRA (Dynamic Optimization for Regularization Adaptation):

	Description: Dynamically adjusts regularization parameters during training to optimize model performance.

	Use Case: Helps in preventing overfitting and improving generalization.


LoReFT (Low-Rank Regularization-based Fine-Tuning):

	Description: Applies low-rank regularization to fine-tune models, reducing the number of parameters.

	Use Case: Reduces computational and memory requirements.


DoReFT (Dynamic Regularization-based Fine-Tuning):

	Description: Dynamically adjusts regularization during fine-tuning to balance performance and efficiency.

	Use Case: Useful for scenarios with varying resource constraints.




###Low-Rank Adaptation Techniques ###


AdaLoRA (Adaptive Low-Rank Adaptation):

	Description: Adapts low-rank components of the model during training to improve efficiency and performance.

	Use Case: Balances model performance with computational efficiency.


ShareLoRA (Shared Low-Rank Adaptation):

	Description: Shares low-rank adaptation layers across tasks or domains to improve generalization.

	Use Case: Useful for multi-task learning and domain adaptation.


QLoRA (Quantization-based Fine-Tuning):

	Description: Combines quantization with low-rank adaptation to further reduce resource requirements.

	Use Case: Ideal for deploying models on resource-constrained devices.


SPLoRA (Sparse Low-Rank Adaptation):

	Description: Incorporates sparsity into low-rank adaptation to reduce the number of parameters.

	Use Case: Enhances computational efficiency and performance.


Qa-LoRA (Quantization and Low-Rank Adaptation):

	Description: Combines quantization and low-rank adaptation to optimize model performance and efficiency.

	Use Case: Suitable for deployment on low-power devices.


LQ-LoRA (Low-Rank and Quantization Adaptation):

	Description: Uses both low-rank and quantization techniques to fine-tune models.

	Use Case: Balances performance with resource constraints.


LoRA-FA (Low-Rank Adaptation with Factor Analysis):

	Description: Uses factor analysis to identify and adapt low-rank components.

	Use Case: Improves the interpretability and efficiency of low-rank adaptation.



### Advanced Adaptation Techniques ###


DyLoRA (Dynamic Low-Rank Adaptation):

	Description: Dynamically adjusts low-rank components during training.

	Use Case: Enhances adaptability and efficiency in dynamic environments.


Intrinsic SAID (Intrinsic Structural and Informational Decomposition):

	Description: Decomposes the model into intrinsic structural and informational components for efficient fine-tuning.

	Use Case: Improves understanding and control over model behavior.


UniPELT (Unified Parameter-Efficient Learning and Transfer):

	Description: A unified framework for parameter-efficient learning and transfer.

	Use Case: Suitable for multi-task learning and domain adaptation.


APT (Adaptive Parameter Tuning)

	Description: Dynamically tunes model parameters during fine-tuning.

	Use Case: Enhances adaptability and performance.


FishMask (Fisher Information Masking):

	Description: Uses Fisher information to identify and fine-tune important parameters.

	Use Case: Enhances model efficiency and performance.


FishDip (Fisher Information with Diverse Initialization Points):

	Description: Combines Fisher information with diverse initialization points to improve fine-tuning.

	Use Case: Enhances model robustness and performance.


SAM (Sharpness-Aware Minimization):

	Description: Minimizes the sharpness of the loss landscape to improve generalization.

	Use Case: Enhances model robustness and performance.



### Fine-Tuning with Specific Focus ###


Bitfit:

	Description: Fine-tunes only the bias terms of the model.

	Use Case: Reduces the number of parameters to be updated, improving efficiency.


S-Bitfit:

	Description: An adaptive version of Bitfit that dynamically selects bias terms to fine-tune.

	Use Case: Enhances performance while maintaining efficiency.


FAR (Feature Augmentation and Regularization):

	Description: Augments features and applies regularization to improve model performance.

	Use Case: Enhances model robustness and generalization.


MAM Adapter (Multi-Adaptive Module Adapter):

	Description: Adapts multiple modules of the model to different tasks or domains.

	Use Case: Suitable for multi-task learning and domain adaptation.


Xattn Tuning (Cross-Attention Tuning):

	Description: Focuses on fine-tuning the cross-attention mechanism in the model.

	Use Case: Improves performance in tasks involving multiple modalities.


NOAH (Neural Optimizer for Adaptive Hyperparameters):

	Description: Adapts hyperparameters during training to optimize performance.

	Use Case: Enhances model training efficiency and performance.


AUTOPEFT (Automated Parameter-Efficient Fine-Tuning):

	Description: Automates the process of parameter-efficient fine-tuning.

	Use Case: Simplifies the fine-tuning process and improves performance.


CIAT (Context-Interdependent Attention Tuning):

	Description: Focuses on tuning attention mechanisms to be context-dependent.

	Use Case: Enhances performance in tasks requiring context-aware processing.


KODA (Knowledge-Oriented Dynamic Adaptation):

	Description: Adapts the model based on knowledge distillation.

	Use Case: Improves performance by leveraging knowledge from other models.


KronA (Kronecker Adaptation):

	Description: Uses Kronecker products to adapt model parameters.

	Use Case: Enhances efficiency and performance in large models.


MerA (Merge and Adapt):

	Description: Merges and adapts model parameters to improve performance.

	Use Case: Suitable for dynamic and adaptive environments.


AdaMIX (Adaptive Mixing):

	Description: Dynamically mixes different model components to improve performance.

	Use Case: Enhances adaptability and robustness.


PHA (Parameter Hierarchy Adaptation):

	Description: Adapts model parameters at different hierarchical levels.

	Use Case: Enhances model efficiency and performance.


DePT (Diverse Parameter Tuning):

	Description: Tunes parameters using diverse initialization points.

	Use Case: Enhances model robustness and performance.


PaFi (Parameter-Free Fine-Tuning):

	Description: Fine-tunes the model without explicitly updating parameters.

	Use Case: Enhances efficiency and performance.


MoSLoRA (Mixture of Experts with Low-Rank Adaptation):

	Description: Combines the mixture of experts with low-rank adaptation.

	Use Case: Suitable for large-scale and diverse tasks.


MOELoRA (Mixture of Experts with Low-Rank Adaptation):

	Description: Similar to MoSLoRA, focusing on combining mixture of experts with low-rank adaptation.

	Use Case: Enhances performance and efficiency in large-scale tasks.




### Community and Ecosystem ###


LoRAHub:

	Description: A community-driven platform for low-rank adaptation techniques.

	Use Case: Provides a hub for sharing and collaborating on low-rank adaptation methods.


KAdaptation (Knowledge-based Adaptation):

	Description: Adapts models based on domain-specific knowledge.

	Use Case: Enhances performance in domain-specific applications.


SPLoRA  (Sparse Low-Rank Adaptation)



Laplace-LoRA (Laplace Distribution Low-Rank Adaptation):






















