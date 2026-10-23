# Weekly Progress Log & GitHub Activity Record

**Student Name:** Arnav Arya  
**Registration Number:** 2430010362  
**Academic Batch:** Batch F  
**Repository:** `MUJ-DS-2430010362`  
**Training Program:** Capstone Project Training Program – Batch F  

---

## Weekly Milestone Breakdown

### Week 1: Project Initialization & Environment Setup
- **Objectives:**
  - Setup GitHub personal repository following standard naming convention `MUJ-DS-2430010362`.
  - Add course instructor as a collaborator.
  - Setup local Python environment with PyTorch and Hugging Face dependencies.
  - Review Stanford Cars dataset literature and fine-grained categorization benchmarks.
- **Key Deliverables:**
  - Initial repository structure created (`README.md`, `requirements.txt`).
  - Google Colab GPU environment verified with CUDA acceleration.
- **Git Activity:** Initial project scaffold commit.

---

### Week 2: Dataset Ingestion & Exploratory Data Analysis (EDA)
- **Objectives:**
  - Programmatically ingest `mteb/StanfordCars` via Hugging Face Datasets API.
  - Verify train/test splits (8,144 train images, 8,041 test images).
  - Inspect sample vehicle images, aspect ratios, and bounding boxes.
  - Extract and verify complete list of 196 class labels spanning vehicle make, model, and year.
- **Key Deliverables:**
  - Visual grid displaying 8 sample car classes (`resources/sample_data_grid.png`).
  - Complete classes dictionary saved to `resources/classes.json` and `resources/classes.txt`.
  - Assignment 1 documentation submitted.
- **Git Activity:** Added exploratory analysis cells and dataset documentation.

---

### Week 3: Data Transformation Pipeline & Baseline Model Exploration
- **Objectives:**
  - Partition training set into 80% training (6,515 images) and 20% validation (1,629 images).
  - Implement data augmentation pipeline: Random Horizontal Flip ($p=0.5$), Random Rotation ($10^\circ$), Resize ($224 \times 224$), and ImageNet Normalization.
  - Design PyTorch DataLoader with batch size 16 and custom collation.
  - Compare classical CNN architectures (ResNet-50) vs. Vision Transformer (ViT-B/16).
- **Key Deliverables:**
  - Reusable data loader module (`code/dataset.py`).
  - ViT-B/16 architecture specification document (`resources/vit_architecture.md`).
- **Git Activity:** Commits implementing dataset transforms and batch loader routines.

---

### Week 4: Model Implementation & Fine-Tuning Execution
- **Objectives:**
  - Implement `StanfordCarsViT` wrapping `google/vit-base-patch16-224` with 196 output classes.
  - Configure AdamW optimizer with learning rate $\eta = 2\times 10^{-5}$ and weight decay $10^{-4}$.
  - Execute 10-epoch fine-tuning on GPU runtime with per-epoch validation.
  - Save best checkpoint based on validation accuracy (`best_vit_stanford_cars.pth`).
- **Key Deliverables:**
  - Best validation accuracy achieved: **76.67%** at Epoch 10.
  - Saved model checkpoint and logged training history to JSON.
  - Assignment 3 documentation submitted.
- **Git Activity:** Commits for model construction, training loop, and checkpoint management.

---

### Week 5: Comprehensive Model Evaluation & Error Analysis
- **Objectives:**
  - Evaluate model on untouched test set (8,041 images).
  - Measure Top-1 Accuracy, Top-5 Accuracy, Weighted Precision, Recall, and F1-Score.
  - Generate full $196 \times 196$ confusion matrix heatmap.
  - Perform class-level analysis to isolate top-10 best and top-10 lowest performing vehicle categories.
- **Key Deliverables:**
  - Test Top-1 Accuracy: **77.02%** | Top-5 Accuracy: **94.80%**.
  - Generated `capstone/results/final_results_summary.csv` and confusion matrix plot.
  - Identified fine-grained confusion patterns (e.g. Audi S4 vs S5).
- **Git Activity:** Added evaluation module (`code/evaluate.py`), metric exports, and plots.

---

### Week 6: Modular Source Code, Capstone Packaging & Final Deliverables
- **Objectives:**
  - Refactor notebook code into modular, production-ready Python package (`code/`).
  - Implement CLI inference tool (`code/predict.py`) for vehicle prediction on external images.
  - Complete technical documentation, presentation summary, and installation manual.
  - Prepare repository for final capstone review and evaluation.
- **Key Deliverables:**
  - Production code in `code/` (`dataset.py`, `model.py`, `train.py`, `evaluate.py`, `predict.py`, `utils.py`).
  - Comprehensive Capstone Report (`capstone/docs/capstone_project_report.md`).
  - Project presentation deck (`presentations/Stanford_Cars_Image_Classification_Capstone.pptx`).
  - Polished `README.md` and `INSTALLATION.md`.
- **Git Activity:** Final capstone integration, documentation updates, and release tag.
