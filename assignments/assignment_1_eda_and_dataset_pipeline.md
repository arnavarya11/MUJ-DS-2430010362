# Assignment 1: Exploratory Data Analysis & PyTorch Transformation Pipeline

**Course / Batch:** Batch F – Capstone Project Training Program  
**Student Name:** Arnav Arya (Reg No: 2430010362)  
**Topic:** Dataset Acquisition, EDA, and Data Pipeline Implementation  

---

## 1. Objective
To systematically acquire the Stanford Cars dataset (`mteb/StanfordCars`), perform thorough exploratory data analysis (EDA), verify class distributions across 196 fine-grained categories, and build a reproducible PyTorch data pipeline with data augmentation.

## 2. Dataset Overview
- **Identifier:** `mteb/StanfordCars`
- **Total Samples:** 16,185 images
  - Training partition: 8,144 images
  - Test partition: 8,041 images
- **Classes:** 196 fine-grained labels containing car make, model, and year (e.g. `Audi R8 Coupe 2012`, `Fiat 500 Abarth 2012`).

## 3. EDA Findings
1. **Resolution Variability:** Raw vehicle images have varied dimensions ranging from $300 \times 200$ to $1200 \times 800$ pixels. A standardized spatial resolution of $224 \times 224$ is required for ViT inputs.
2. **Aspect Ratios:** Predominantly landscape orientation, typical of automotive photography.
3. **Class Balance:** Approximately 24 to 68 images per category across train and test sets, reflecting balanced representation without severe class starvation.

## 4. Pipeline Implementation
An 80/20 train/validation split was created from the original 8,144 training samples:
- Training: 6,515 images
- Validation: 1,629 images

Transformations applied:
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```
DataLoaders were configured with batch size 16 to maximize GPU tensor core utilization.
