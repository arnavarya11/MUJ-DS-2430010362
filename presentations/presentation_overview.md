# Capstone Project Presentation Overview

**Presentation File:** [`Stanford_Cars_Image_Classification_Capstone.pptx`](Stanford_Cars_Image_Classification_Capstone.pptx)  
**Author:** Arnav Arya | Registration Number: 2430010362  
**Branch:** B.Tech Computer Science Engineering  
**Batch:** Batch F  
**Program:** Capstone Project Training Program  

---

## Slide-by-Slide Executive Transcript

### Slide 1: Title & Student Details
- **Project Title:** Stanford Cars Image Classification Using Deep Learning
- **Subtitle:** A deep learning workflow for recognizing car categories from images
- **Student Details:** Arnav Arya (Reg. No. 2430010362)
- **Academic Batch:** Batch F – B.Tech Computer Science Engineering
- **GitHub Repository:** [arnavarya11/MUJ-DS-2430010362](https://github.com/arnavarya11/MUJ-DS-2430010362)

### Slide 2: Problem Statement & Motivation
- **The Challenge:** Fine-grained visual categorization (FGVC) involves separating vehicles across 196 subtle makes, models, and production years.
- **Why It Matters:** Essential for intelligent traffic management, automated toll collection, smart parking, insurance claims verification, and autonomous driving context.

### Slide 3: Dataset at a Glance
- **Identifier:** `mtb/StanfordCars` (Stanford AI Lab benchmark)
- **Scale:** 16,185 total images across 196 fine-grained vehicle classes
- **Splits:** 6,515 training images (80%), 1,629 validation images (20%), 8,041 test images
- **Attributes:** Variable image resolutions, diverse lighting conditions, real-world angles, and complex backgrounds

### Slide 4: Technology Stack
- **Frameworks:** Python 3.10+, PyTorch 2.x, Hugging Face Datasets & Transformers
- **Compute:** Google Colab GPU environment (NVIDIA T4 / A100)
- **Evaluation Tools:** Scikit-learn, Pandas, Matplotlib, Seaborn

### Slide 5: Methodology & Proposed Workflow
1. **Data Ingestion:** Load streaming dataset from Hugging Face.
2. **Preprocessing & Augmentation:** Random rotations (10°), horizontal flips, ImageNet normalization ($224 \times 224$).
3. **Model Architecture:** Vision Transformer (`google/vit-base-patch16-224`) with fine-tuned linear classification head.
4. **Optimization:** AdamW optimizer ($\text{lr} = 2\times 10^{-5}$, $\text{weight decay} = 10^{-4}$), Cross-Entropy Loss.
5. **Evaluation:** Top-1 / Top-5 classification metrics, confusion matrix, error analysis.

### Slide 6: Vision Transformer (ViT-B/16) Architecture
- **Patch Embedding:** Flattens image into $14 \times 14 = 196$ patches of $16 \times 16$ pixels.
- **Self-Attention:** 12 Transformer layers with 12 attention heads capture global relationships across distant vehicle features.
- **Classification Head:** Linear projection from 768-dim `[CLS]` token to 196 vehicle classes.

### Slide 7: Training Progress & Convergence
- **Epochs:** 10 training epochs with continuous checkpointing.
- **Loss Progression:** Training loss decreased from $4.9415 \rightarrow 0.3146$; validation loss dropped to $1.0938$.
- **Validation Accuracy:** Steady rise from $8.84\% \rightarrow 76.67\%$ (Best model checkpoint saved at Epoch 10).

### Slide 8: Quantitative Evaluation & Results
- **Test Top-1 Accuracy:** **77.02%**
- **Test Top-5 Accuracy:** **94.80%**
- **Weighted Precision:** **78.10%**
- **Weighted Recall:** **77.02%**
- **Weighted F1-Score:** **76.84%**
- **Macro F1-Score:** **76.78%**

### Slide 9: Qualitative Visual Analysis & Demo Predictions
- **High-Performing Vehicles:** Distinctive body profiles (Hummer H1 SUV: 0.99 F1, Dodge Caravan: 0.99 F1, Fiat 500: 0.98 F1).
- **Confusion Hotspots:** Subtle differences between adjacent year models (e.g., Audi S4 vs. Audi S5; Chevrolet Express Van vs. Cargo Van).
- **Inference Verification:** Confirmed predictions on real-world car images (Audi R8: 69.8% confidence; Mini Cooper: 31.8% top-1).

### Slide 10: Conclusion & Future Enhancements
- **Summary:** Successfully demonstrated high-precision fine-grained vehicle recognition using Vision Transformers.
- **Future Directions:**
  - Exploring Swin Transformer or ViT-Large architectures
  - Part-based bounding box attention for bumper/grille localization
  - Quantization and ONNX runtime export for edge deployment on smart camera systems
