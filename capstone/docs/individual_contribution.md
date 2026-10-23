# Individual Contribution Statement

**Student Name:** Arnav Arya  
**Registration Number:** 2430010362  
**Academic Program / Branch:** B.Tech Computer Science Engineering  
**Academic Batch:** Batch F  
**Capstone Project Title:** Fine-Grained Vehicle Classification Using Vision Transformers  
**Repository:** [arnavarya11/MUJ-DS-2430010362](https://github.com/arnavarya11/MUJ-DS-2430010362)  

---

## 1. Summary of Contributions
In accordance with **Step 8** of the *Capstone Project Guidelines & Submission Process*, this document provides an itemized breakdown of the individual technical contributions made throughout the design, implementation, evaluation, and documentation of this capstone project.

```
┌────────────────────────────────────────────────────────────────────────┐
│                   ARNAV ARYA - INDIVIDUAL OWNERSHIP                    │
├──────────────────────────┬─────────────────────────────────────────────┤
│ Area                     │ Key Responsibility & Deliverable            │
├──────────────────────────┼─────────────────────────────────────────────┤
│ 1. Data Engineering      │ HuggingFace dataset integration & transforms│
│ 2. Model Development     │ ViT-B/16 architecture & transfer learning   │
│ 3. Training & Tuning     │ AdamW optimizer, loss monitoring, checkpoints│
│ 4. Evaluation & Testing  │ 196-class metrics, confusion matrix, errors │
│ 5. Codebase Engineering  │ Modular `code/` package & CLI inference tool│
│ 6. Documentation         │ Technical reports, installation & slides    │
└──────────────────────────┴─────────────────────────────────────────────┘
```

---

## 2. Detailed Breakdown of Technical Areas

### A. Data Engineering & Collection
- **Dataset Integration:** Configured automated loading of `mteb/StanfordCars` containing 16,185 annotated vehicle images.
- **Data Partitioning:** Implemented reproducible 80/20 train/validation splits (6,515 train images, 1,629 validation images) with fixed random seed (42).
- **Augmentation Pipeline:** Engineered robust torchvision transformation pipeline with `RandomHorizontalFlip(p=0.5)`, `RandomRotation(degrees=10)`, and spatial resizing to $224 \times 224$ pixels.
- **ImageNet Normalization:** Formulated mean/standard-deviation channel normalization ($[0.485, 0.456, 0.406]$, $[0.229, 0.224, 0.225]$) matching ViT pretraining standards.

### B. Model Development & Architecture
- **Architecture Selection:** Selected Vision Transformer (`google/vit-base-patch16-224`) based on comparative trade-off analysis against convolutional architectures (ResNet-50).
- **Head Adaptation:** Adapted classification head from 1000 ImageNet classes to 196 fine-grained vehicle categories.
- **Layer Exploration:** Evaluated full fine-tuning vs. head-only freezing, opting for end-to-end low learning rate fine-tuning to adapt high-level attention heads to vehicle silhouettes and badges.

### C. Training Execution & Checkpointing
- **Optimization Strategy:** Implemented AdamW optimizer with decoupled weight decay ($\text{lr} = 2 \times 10^{-5}$, $\text{weight decay} = 10^{-4}$).
- **Validation Loop:** Developed per-epoch validation loop to monitor generalization gap and prevent overfitting.
- **Checkpoint System:** Created model serialization mechanism saving `best_vit_stanford_cars.pth` on validation accuracy improvements, achieving peak validation accuracy of **76.67%** at Epoch 10.

### D. Testing, Evaluation & Error Analysis
- **Test Set Benchmark:** Executed inference on 8,041 test images, verifying **77.02% Top-1 Accuracy** and **94.80% Top-5 Accuracy**.
- **Comprehensive Metrics:** Computed Weighted Precision (78.10%), Recall (77.02%), F1-Score (76.84%), and Macro F1 (76.78%).
- **Error Diagnostics:** Generated $196 \times 196$ confusion matrix and isolated root causes of confusion between adjacent model years and badge variants.

### E. Code Modularization & CLI Deployment
- **Refactoring:** Refactored experimental Colab notebook code into a clean, reusable Python package under `code/`:
  - `dataset.py`: Data loaders and augmentation logic
  - `model.py`: ViT model instantiation and configuration
  - `train.py`: CLI training script with argument parsing
  - `evaluate.py`: CLI evaluation script generating test metrics and matrices
  - `predict.py`: Standalone CLI inference tool with visualization capabilities
  - `utils.py`: Metric calculation and plotting helpers
- **Testing & Verification:** Verified module imports and execution across virtual environment and Colab runtimes.

### F. Technical Documentation & Presentation
- Authored the main `README.md`, `INSTALLATION.md`, and technical specifications.
- Authored weekly milestone reports in `assignments/`.
- Prepared slide-by-slide executive summary for `Stanford_Cars_Image_Classification_Capstone.pptx`.
- Compiled formal technical capstone project report.

---

## 3. GitHub Activity & Workflow Compliance
- **Regular Commits:** Maintained meaningful, staged commits tracking the project's chronological progression.
- **Collaborator Management:** Added course instructor to personal repository.
- **Milestone & Issue Tracking:** Formulated GitHub Issues and Pull Request workflows covering Dataset Collection, Model Development, Testing, Documentation, and Deployment.
