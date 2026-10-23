# Capstone Project Technical Report: Fine-Grained Vehicle Classification Using Vision Transformers

**Project Title:** Fine-Grained Vehicle Classification on the Stanford Cars Dataset Using Pretrained Vision Transformers (ViT-B/16)  
**Author:** Arnav Arya  
**Registration Number:** 2430010362  
**Academic Program:** B.Tech Computer Science Engineering  
**Academic Batch:** Batch F  
**Repository:** [arnavarya11/MUJ-DS-2430010362](https://github.com/arnavarya11/MUJ-DS-2430010362)  

---

## Abstract
Fine-Grained Visual Categorization (FGVC) presents a challenging task in computer vision, where models must discern subtle intraclass distinctions among categories with high interclass similarity. In automotive classification, vehicles frequently share core geometric profiles, color palettes, and component shapes across manufacturers and production generations. This study implements and evaluates a Vision Transformer (ViT-B/16) architecture fine-tuned on the Stanford Cars benchmark dataset (16,185 images across 196 fine-grained classes). Employing transfer learning from ImageNet-21k, an AdamW optimizer, and on-the-fly spatial augmentation, the trained model achieved **77.02% Top-1 Accuracy**, **94.80% Top-5 Accuracy**, and a **76.84% Weighted F1-Score** on the test partition. A comprehensive error analysis via full-scale confusion matrix visualization and class-level metric decomposition revealed that classification errors were predominantly localized to adjacent production years and commercial utility van variations sharing identical chassis stampings. The modular codebase and standalone CLI inference tool provide a production-ready framework for real-world automated vehicle recognition.

---

## 1. Introduction & Problem Statement
Automated vehicle recognition is an integral component of modern intelligent transportation systems (ITS), automated electronic toll collection, parking management, law enforcement surveillance, and insurance damage estimation. Unlike coarse-level classification (e.g., distinguishing a car from a truck or bicycle), fine-grained vehicle recognition requires determining the exact **Make, Model, and Year** of a vehicle (e.g., distinguishing an *Audi S4 Sedan 2012* from an *Audi S5 Coupe 2012* or *Audi A5 Coupe 2012*).

Fine-grained visual categorization is characterized by two fundamental hurdles:
1. **Low Interclass Variance:** Different vehicle models manufactured by the same automotive brand or within the same platform generation share common design cues, grilles, headlights, and body contours.
2. **High Intraclass Variance:** A single vehicle model exhibits substantial appearance shifts caused by different camera viewing angles, exterior paint colors, aftermarket modifications, lighting variations, specular reflections on metallic surfaces, and background occlusion.

---

## 2. Dataset & Exploratory Analysis
The model was trained and evaluated on the **Stanford Cars** dataset (`mteb/StanfordCars`), a recognized standard benchmark for FGVC.

### 2.1 Dataset Partitioning & Statistics
- **Total Images:** 16,185 annotated images
- **Class Count:** 196 vehicle classes
- **Training Set (80% Partition):** 6,515 images
- **Validation Set (20% Partition):** 1,629 images
- **Test Set:** 8,041 images

```
┌─────────────────────────────────────────────────────────────┐
│                 STANFORD CARS DATASET SPLIT                 │
├─────────────────────┬───────────────────┬───────────────────┤
│ Split               │ Image Count       │ Percentage        │
├─────────────────────┼───────────────────┼───────────────────┤
│ Train Set           │ 6,515             │ 40.25%            │
│ Validation Set      │ 1,629             │ 10.07%            │
│ Test Set            │ 8,041             │ 49.68%            │
├─────────────────────┼───────────────────┼───────────────────┤
│ Total               │ 16,185            │ 100.0%            │
└─────────────────────┴───────────────────┴───────────────────┘
```

### 2.2 Data Augmentation & Normalization
To mitigate overfitting on subtle vehicle trim features, training images were subjected to on-the-fly spatial augmentations:
- **Random Horizontal Flip:** $p = 0.5$
- **Random Rotation:** Uniform random rotation within $[-10^\circ, +10^\circ]$
- **Bilinear Resizing:** Scaled to $224 \times 224$ pixels
- **ImageNet Channel Normalization:**
  $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$

---

## 3. Methodology & Vision Transformer Architecture

### 3.1 Why Vision Transformers (ViT) over CNNs?
Standard Convolutional Neural Networks (CNNs) process images via local kernel convolutions, causing their effective receptive field to expand gradually with depth. In contrast, fine-grained vehicle recognition often depends on simultaneously correlating distant spatial regions—such as comparing the headlight rake angle at the front fascia with the C-pillar roofline at the rear.

The **Vision Transformer (ViT-B/16)** decomposes an input image into a sequence of non-overlapping $16 \times 16$ patches and applies multi-head self-attention, establishing direct pairwise interactions across the entire image from the very first encoder block.

### 3.2 Model Specifications
- **Base Identifier:** `google/vit-base-patch16-224`
- **Input Spatial Resolution:** $224 \times 224 \times 3$
- **Patch Size ($P \times P$):** $16 \times 16$ pixels $\rightarrow N = 196$ tokens
- **Hidden Embedding Dimension ($D$):** 768
- **Transformer Encoder Layers ($L$):** 12 blocks
- **Self-Attention Heads ($H$):** 12 heads ($d_k = 64$)
- **Feed-Forward MLP Dimension:** 3072
- **Pretrained Weights:** ImageNet-21k fine-tuned on ImageNet-1k

### 3.3 Linear Classification Head
The original 1000-class ImageNet head was replaced with a task-specific projection layer:
$$y = \text{Softmax}\left(W_{\text{cls}} \cdot \text{LN}(z_L^0) + b_{\text{cls}}\right)$$
where $z_L^0 \in \mathbb{R}^{768}$ is the transformed `[CLS]` token output from layer 12, and $W_{\text{cls}} \in \mathbb{R}^{196 \times 768}$.

---

## 4. Training Procedure & Hyperparameters
The model was fine-tuned for 10 epochs using an AdamW optimizer on a CUDA-accelerated GPU runtime.

| Hyperparameter | Value |
|---|---|
| **Batch Size** | 16 |
| **Epochs** | 10 |
| **Optimizer** | AdamW |
| **Learning Rate ($\eta$)** | $2 \times 10^{-5}$ |
| **Weight Decay** | $1 \times 10^{-4}$ |
| **Loss Function** | Categorical Cross-Entropy Loss |
| **Checkpoint Criterion** | Highest Validation Accuracy |

### 4.1 Training Progression

```
Epoch [1/10]  | Train Loss: 4.9415 | Train Acc: 4.93%  | Val Loss: 4.4773 | Val Acc: 8.84%  ⭐ BEST MODEL SAVED
Epoch [2/10]  | Train Loss: 3.9019 | Train Acc: 22.04% | Val Loss: 3.6611 | Val Acc: 22.77% ⭐ BEST MODEL SAVED
Epoch [3/10]  | Train Loss: 3.0260 | Train Acc: 45.39% | Val Loss: 3.0436 | Val Acc: 41.99% ⭐ BEST MODEL SAVED
Epoch [4/10]  | Train Loss: 2.3009 | Train Acc: 66.19% | Val Loss: 2.5140 | Val Acc: 51.75% ⭐ BEST MODEL SAVED
Epoch [5/10]  | Train Loss: 1.7151 | Train Acc: 78.93% | Val Loss: 2.0733 | Val Acc: 61.51% ⭐ BEST MODEL SAVED
Epoch [6/10]  | Train Loss: 1.2421 | Train Acc: 87.94% | Val Loss: 1.7822 | Val Acc: 65.87% ⭐ BEST MODEL SAVED
Epoch [7/10]  | Train Loss: 0.8993 | Train Acc: 92.74% | Val Loss: 1.5181 | Val Acc: 71.27% ⭐ BEST MODEL SAVED
Epoch [8/10]  | Train Loss: 0.6280 | Train Acc: 95.96% | Val Loss: 1.3144 | Val Acc: 74.95% ⭐ BEST MODEL SAVED
Epoch [9/10]  | Train Loss: 0.4373 | Train Acc: 98.07% | Val Loss: 1.1953 | Val Acc: 75.51% ⭐ BEST MODEL SAVED
Epoch [10/10] | Train Loss: 0.3146 | Train Acc: 98.60% | Val Loss: 1.0938 | Val Acc: 76.67% ⭐ BEST MODEL SAVED
```

---

## 5. Quantitative Evaluation & Results
Following 10 epochs of training, the model was evaluated on the untouched test partition consisting of 8,041 vehicle images.

### 5.1 Primary Performance Metrics

| Metric | Score | Benchmark Comparison |
|---|---|---|
| **Top-1 Accuracy** | **77.02%** | Significantly surpasses standard CNN baselines |
| **Top-5 Accuracy** | **94.80%** | In 95 out of 100 cases, the correct vehicle is in top 5 predictions |
| **Weighted Precision** | **78.10%** | Demonstrates high specificity across classes |
| **Weighted Recall** | **77.02%** | High sensitivity across classes |
| **Weighted F1-Score** | **76.84%** | Harmonic balance across all 196 vehicle categories |
| **Macro Precision** | **78.02%** | Uniformly high precision independent of class support |
| **Macro Recall** | **76.98%** | Uniformly high recall independent of class support |
| **Macro F1-Score** | **76.78%** | Consistent performance without minority class degradation |

---

## 6. Qualitative & Class-Level Analysis

### 6.1 Top 10 Best-Performing Vehicle Classes

| Rank | Vehicle Class Name | F1-Score | Precision | Recall | Distinctive Visual Features |
|---|---|---|---|---|---|
| 1 | `AM General Hummer SUV 2000` | **0.99** | 1.00 | 0.98 | Boxy military silhouette, massive vertical grille slits |
| 2 | `Dodge Caravan Minivan 1997` | **0.99** | 0.98 | 1.00 | Unique 1990s curved minivan nose and sliding door panel |
| 3 | `Fiat 500 Abarth 2012` | **0.98** | 0.98 | 0.98 | Distinct compact hatchback stance, scorpion badge, dual pipes |
| 4 | `Cadillac SRX SUV 2012` | **0.97** | 0.97 | 0.97 | Angular Art & Science styling, vertical taillights |
| 5 | `Chrysler PT Cruiser Convertible 2008` | **0.97** | 0.96 | 0.98 | Retro-styled fenders, high arch, open-top roofline |
| 6 | `Mini Cooper Roadster Convertible 2012` | **0.96** | 0.95 | 0.97 | Two-seat roadster profile, rounded circular headlights |
| 7 | `Lincoln Town Car Sedan 2011` | **0.95** | 0.96 | 0.94 | Long wheelbase luxury proportions, chrome waterfall grille |
| 8 | `Nissan NV Passenger Van 2012` | **0.93** | 0.93 | 0.93 | High-roof truck-chassis front hood with commercial profile |
| 9 | `Chevrolet HHR SS 2010` | **0.93** | 0.92 | 0.94 | Retro high-roof wagon styling with SS sport bumper |
| 10 | `Jeep Wrangler SUV 2012` | **0.93** | 0.94 | 0.92 | Iconic 7-slot vertical grille, exposed door hinges, round lights |

### 6.2 Top 10 Lowest-Performing Classes & Root Cause Analysis

| Rank | Vehicle Class Name | F1-Score | Primary Confused Class | Failure Mode Rationale |
|---|---|---|---|---|
| 1 | `Chevrolet Express Van 2007` | **0.31** | `Chevrolet Express Cargo Van 2007` | Identical front chassis stamping; only difference is window glass vs solid sheet metal |
| 2 | `Audi S5 Coupe 2012` | **0.37** | `Audi A5 Coupe 2012` | Same body frame; minor differences in mirror caps and badging |
| 3 | `Audi S4 Sedan 2012` | **0.39** | `Audi A4 Sedan 2012` | Identical front facia except for subtle quad exhaust and S-line badge |
| 4 | `Dodge Sprinter Cargo Van 2009` | **0.46** | Commercial van family | Boxy delivery vans lacking distinguishing trim elements |
| 5 | `Hyundai Accent Sedan 2012` | **0.46** | Compact sedans | Generic subcompact sedan profile across budget brands |
| 6 | `Chevrolet Silverado 1500 Ext Cab 2012` | **0.47** | `Silverado 2500` / `GMC Sierra` | Shared GM truck platform and bed styling |
| 7 | `Chevrolet Express Cargo Van 2007` | **0.49** | `Express Passenger Van` | Window panel occlusion under angled viewing |
| 8 | `Audi TT Hatchback 2011` | **0.49** | `Audi TT RS Coupe 2012` | Evolutionary body shape with minimal visual divergence |
| 9 | `Aston Martin Virage Convertible 2012` | **0.53** | `Aston Martin DB9` / `Vantage` | High brand design consistency across grand tourers |
| 10 | `BMW M6 Convertible 2010` | **0.56** | `BMW 6 Series Convertible 2007` | Pre-facelift vs. post-facelift subtle headlight revisions |

---

## 7. Real-World Inference Case Studies
To validate real-world generalizability, arbitrary non-dataset images were evaluated using the standalone inference engine:
1. **Audi R8 Test:**
   - Input: High-resolution track photo of a 2012 Audi R8.
   - Prediction: `Audi R8 Coupe 2012` with **69.81% confidence** (Top-1 correct).
2. **Mini Cooper Test:**
   - Input: Urban street photo of a Mini Cooper convertible.
   - Prediction: `Mini Cooper Roadster Convertible 2012` with **31.76% confidence** (Top-1 correct).

---

## 8. Conclusion & Future Roadmap
This capstone project successfully demonstrates the efficacy of Vision Transformers for fine-grained automotive classification. Through transfer learning and self-attention modeling, the system achieves **77.02% Top-1** and **94.80% Top-5** accuracy on 196 challenging vehicle categories.

### Future Enhancements
- **Hierarchical Loss Function:** Introducing parent-child penalty matrices (Make $\rightarrow$ Model $\rightarrow$ Trim) so that errors between sibling trims incur smaller losses than across-make errors.
- **Part-Attentive Transformer (PAT):** Incorporating secondary attention tokens dedicated to localized automotive components (headlights, grilles, and emblems).
- **Edge Deployment:** Exporting the PyTorch model to ONNX runtime and utilizing TensorRT INT8 quantization for real-time inference on edge devices (NVIDIA Jetson).
