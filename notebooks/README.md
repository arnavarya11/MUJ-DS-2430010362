# Jupyter Notebooks Guide

This directory contains the experimental and exploratory Jupyter notebooks developed for the Stanford Cars Vision Transformer Capstone Project.

## Primary Notebook
- **File:** [`2430010362.ipynb`](2430010362.ipynb)
- **Author:** Arnav Arya (Reg No: 2430010362)
- **Batch:** Batch F

## Notebook Content & Execution Sequence
The notebook is systematically structured across 62 execution cells:
1. **Environment Setup & GPU Check:** Verification of PyTorch, CUDA device detection, dependency installations.
2. **Dataset Acquisition:** Loading `mteb/StanfordCars` via Hugging Face Datasets API.
3. **Exploratory Data Analysis (EDA):** Visualizing random vehicle samples, checking resolutions, analyzing class distributions across 196 categories.
4. **Data Preprocessing & Augmentation:** Stratified 80/20 train-validation splitting, torchvision data transformation pipeline (`RandomHorizontalFlip`, `RandomRotation(10)`).
5. **DataLoader Construction:** Batch size configuration (`batch_size=16`) and shape validation (`[16, 3, 224, 224]`).
6. **ViT-B/16 Initialization:** Loading `google/vit-base-patch16-224` with 196 classification heads and inspecting layer parameters.
7. **Training & Validation Loop:** 10-epoch training with AdamW optimizer, tracking loss and accuracy, checkpointing the best model weights.
8. **Test Set Evaluation:** Computing Top-1 Accuracy (77.02%), Top-5 Accuracy (94.80%), Weighted & Macro F1 metrics.
9. **Visual Analytics & Plots:** Plotting loss curves, accuracy curves, 196x196 confusion matrix heatmap, and best/worst performing vehicle classes.
10. **Inference & Demo:** Uploading sample external images (e.g., Audi R8, Mini Cooper) and visualizing top-5 prediction bar charts.
11. **Export Artifacts:** Saving JSON metrics, configurations, and CSV reports to Google Drive/local storage.

## How to Run

### Option A: Google Colab (Recommended)
1. Upload `2430010362.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Select **Runtime** → **Change runtime type** → **T4 GPU** (or any available GPU).
3. Execute all cells from top to bottom.

### Option B: Local Jupyter Lab / VS Code
```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Launch Jupyter
jupyter notebook notebooks/2430010362.ipynb
```
Ensure you have installed all packages specified in `requirements.txt`.
