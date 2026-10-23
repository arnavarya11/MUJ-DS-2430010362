# Assignment 4: Comprehensive Model Evaluation & Error Diagnostics

**Course / Batch:** Batch F – Capstone Project Training Program  
**Student Name:** Arnav Arya (Reg No: 2430010362)  
**Topic:** Quantitative Metrics, Confusion Matrix, and Per-Class Error Analysis  

---

## 1. Objective
To rigorously evaluate the trained Vision Transformer model on the complete, untouched test set (8,041 images), measure Top-1 and Top-5 recognition rates, analyze class-level precision and recall, and diagnose failure modes.

## 2. Test Set Performance Summary

| Metric | Score | Interpretation |
|---|---|---|
| **Top-1 Accuracy** | **77.02%** | Correct exact make, model, and year prediction |
| **Top-5 Accuracy** | **94.80%** | True vehicle is within top 5 model candidates |
| **Weighted Precision** | **78.10%** | High positive predictive value across classes |
| **Weighted Recall** | **77.02%** | High sensitivity across classes |
| **Weighted F1-Score** | **76.84%** | Harmonic balance between precision and recall |
| **Macro F1-Score** | **76.78%** | Uniformly high performance across all 196 categories |

## 3. High-Performing vs. Low-Performing Classes

### Top-3 Best Performing Classes
1. `AM General Hummer SUV 2000` (F1: 0.99) — Highly unique boxy silhouette and front grille.
2. `Dodge Caravan Minivan 1997` (F1: 0.99) — Distinctive 90s minivan curvature and proportions.
3. `Fiat 500 Abarth 2012` (F1: 0.98) — Compact subcompact profile with specific dual exhaust and badging.

### Top-3 Most Challenging Classes
1. `Chevrolet Express Van 2007` (F1: 0.31) — Readily confused with `Chevrolet Express Cargo Van 2007` due to near-identical exterior chassis.
2. `Audi S5 Coupe 2012` (F1: 0.37) — Frequently confused with `Audi A5 Coupe 2012` and `Audi S4 Sedan 2012`.
3. `Audi S4 Sedan 2012` (F1: 0.39) — Subtle visual divergence from other mid-size Audi sedans of the same production era.

## 4. Remediation Recommendations
- Implement multi-scale patch attention or dual-branch architecture focusing on fine-grained regions (e.g. badges, front bumper splitters).
- Incorporate vehicle brand/manufacturer hierarchy into the loss function (Hierarchical Cross-Entropy).
