# Assignment 2: Baseline Comparison & Vision Transformer Architecture

**Course / Batch:** Batch F – Capstone Project Training Program  
**Student Name:** Arnav Arya (Reg No: 2430010362)  
**Topic:** Model Architecture Exploration and Vision Transformer Formulation  

---

## 1. Objective
To evaluate model architectural options for Fine-Grained Visual Categorization (FGVC) on the Stanford Cars dataset and formulate the Vision Transformer (ViT-B/16) model architecture with transfer learning.

## 2. Comparative Architectural Analysis

| Dimension | Classical CNN (e.g. ResNet-50) | Vision Transformer (ViT-B/16) |
|---|---|---|
| **Inductive Bias** | Strong local inductive bias (locality & translation invariance) | Minimal inductive bias; learns global patterns directly |
| **Receptive Field** | Grows linearly with convolutional network depth | Global receptive field from Layer 1 via Self-Attention |
| **Feature Correlation** | Local kernel convolutions | All-to-all patch attention across the vehicle |
| **Fine-Grained Suitability** | May miss distant co-occurrences (e.g. grille to roof shape) | Directly links front fascia, wheel profile, and rear contours |

## 3. ViT-B/16 Model Configuration
- **Base Checkpoint:** `google/vit-base-patch16-224`
- **Patch Resolution:** $16 \times 16$ pixels $\rightarrow 196$ patch tokens
- **Hidden Embedding:** 768 dimensions
- **Transformer Encoder Blocks:** 12 layers with 12 multi-head attention heads
- **Classification Head:** Linear projection $768 \rightarrow 196$ vehicle classes
- **Initial Weights:** Pretrained on ImageNet-21k, fine-tuned on ImageNet-1k

## 4. Transfer Learning Strategy
The Transformer encoder layers are initialized with pretrained weights, capturing foundational visual primitives (edges, textures, metallic specular reflections, geometric contours). The classifier head is replaced with a 196-way linear layer with random initialization and trained end-to-end with a low learning rate ($\eta = 2 \times 10^{-5}$).
