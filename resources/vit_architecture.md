# Vision Transformer (ViT-B/16) Architecture Reference

## 1. Background & Motivation
Convolutional Neural Networks (CNNs) rely on local inductive biases (translation equivariance and locality). While effective for standard classification, fine-grained visual classification (FGVC) requires modeling long-range contextual relationships and global dependencies between distant vehicle components (e.g., front headlights, side trim lines, roof curvature, and rear spoilers).

The **Vision Transformer (ViT)** (Dosovitskiy et al., 2020) adapts standard Transformer encoders directly to computer vision without convolutional operators, learning global self-attention across image patches.

## 2. ViT-B/16 Structural Specifications

| Parameter | Value |
|---|---|
| **Base Model Identifier** | `google/vit-base-patch16-224` |
| **Input Image Resolution** | $224 \times 224 \times 3$ |
| **Patch Size ($P \times P$)** | $16 \times 16$ pixels |
| **Number of Patches ($N$)** | $(224 / 16) \times (224 / 16) = 14 \times 14 = 196$ patches |
| **Embedding Dimension ($D$)** | 768 |
| **Transformer Encoder Layers ($L$)** | 12 |
| **Multi-Head Self-Attention Heads ($H$)** | 12 (each head dimension $d_k = 64$) |
| **MLP Hidden Dimension** | 3072 |
| **Classifier Head** | Linear layer mapping $768 \rightarrow 196$ vehicle classes |
| **Pretrained Weights** | ImageNet-21k (14M images, 21,843 classes) fine-tuned on ImageNet 1k |

## 3. Mathematical Formulation

### Patch Projection & Embedding
Given an input image $x \in \mathbb{R}^{H \times W \times C}$, it is flattened into a sequence of 2D patches $x_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$:
$$z_0 = [x_{\text{class}}; x_p^1 E; x_p^2 E; \dots; x_p^N E] + E_{\text{pos}}$$
where:
- $E \in \mathbb{R}^{(P^2 \cdot C) \times D}$ is the linear patch projection matrix.
- $x_{\text{class}}$ is the learnable classification `[CLS]` token prepended to the sequence.
- $E_{\text{pos}} \in \mathbb{R}^{(N + 1) \times D}$ represents 1D learnable positional embeddings.

### Multi-Head Self-Attention (MSA)
For each transformer block $l \in [1, L]$:
$$z'_l = \text{MSA}(\text{LN}(z_{l-1})) + z_{l-1}$$
$$z_l = \text{MLP}(\text{LN}(z'_l)) + z'_l$$

Self-attention computes dynamic attention weights between all patch pairs:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

### Classification Head
The state of the `[CLS]` token at the output of the 12th layer serves as the aggregate representation:
$$y = \text{Softmax}(W_{\text{head}} \cdot \text{LN}(z_L^0) + b_{\text{head}})$$
where $W_{\text{head}} \in \mathbb{R}^{196 \times 768}$.

## 4. Why ViT-B/16 for Stanford Cars?
1. **Global Receptive Field from Layer 1:** ViT models cross-patch interactions instantly, comparing front fascia cues directly with cabin proportion.
2. **Transfer Learning Efficiency:** Pretrained ImageNet representations provide robust edge, texture, and component feature detectors, achieving **77.02% Top-1** and **94.80% Top-5** accuracy within 10 epochs.
