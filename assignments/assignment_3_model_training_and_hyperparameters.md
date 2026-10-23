# Assignment 3: Model Training, Hyperparameter Tuning & Convergence

**Course / Batch:** Batch F – Capstone Project Training Program  
**Student Name:** Arnav Arya (Reg No: 2430010362)  
**Topic:** Model Training Execution, Loss Tracking, and Checkpoint Selection  

---

## 1. Objective
To implement an end-to-end training pipeline for the ViT-B/16 model on the Stanford Cars training split, monitor loss and accuracy trajectories, and save optimal checkpoint weights based on validation accuracy.

## 2. Experimental Hyperparameters

| Hyperparameter | Value | Rationale |
|---|---|---|
| **Optimizer** | AdamW | Decoupled weight decay provides regularized convergence |
| **Learning Rate ($\eta$)** | $2 \times 10^{-5}$ | Preserves pretrained transformer representations without catastrophic forgetting |
| **Weight Decay** | $1 \times 10^{-4}$ | Prevents overfitting on subtle vehicle trim features |
| **Batch Size** | 16 | Optimal balance between gradient stability and GPU memory fit |
| **Epochs** | 10 | Sufficient for complete fine-tuning convergence |
| **Loss Function** | Cross-Entropy Loss | Standard multi-class classification criterion |

## 3. Training & Validation Epoch Trajectory

| Epoch | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) | Checkpoint Status |
|---|---|---|---|---|---|
| 1 | 4.9415 | 4.93% | 4.4773 | 8.84% | Saved (Best: 8.84%) |
| 2 | 3.9019 | 22.04% | 3.6611 | 22.77% | Saved (Best: 22.77%) |
| 3 | 3.0260 | 45.39% | 3.0436 | 41.99% | Saved (Best: 41.99%) |
| 4 | 2.3009 | 66.19% | 2.5140 | 51.75% | Saved (Best: 51.75%) |
| 5 | 1.7151 | 78.93% | 2.0733 | 61.51% | Saved (Best: 61.51%) |
| 6 | 1.2421 | 87.94% | 1.7822 | 65.87% | Saved (Best: 65.87%) |
| 7 | 0.8993 | 92.74% | 1.5181 | 71.27% | Saved (Best: 71.27%) |
| 8 | 0.6280 | 95.96% | 1.3144 | 74.95% | Saved (Best: 74.95%) |
| 9 | 0.4373 | 98.07% | 1.1953 | 75.51% | Saved (Best: 75.51%) |
| 10 | 0.3146 | 98.60% | 1.0938 | 76.67% | Saved (Best: 76.67%) |

## 4. Key Takeaways
- The model achieved smooth, monotonic convergence across all 10 epochs.
- Validation loss declined steadily from $4.4773 \rightarrow 1.0938$.
- Best checkpoint was saved at Epoch 10 with **76.67% validation accuracy**.
