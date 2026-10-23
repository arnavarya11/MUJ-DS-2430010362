# GitHub Workflow, Issues & Collaborator Guide

This guide provides instructions and ready-to-use templates for fulfilling **Steps 3, 7, 9, 10, and 12** of the Capstone Project Guidelines.

---

## 1. Step 3: Add Instructor as Collaborator
Under **Step 3**, you must add your instructor's GitHub account as a collaborator to your personal repository:
1. Navigate to your repository on GitHub: `https://github.com/arnavarya11/MUJ-DS-2430010362`
2. Click on **Settings** (top right repository tab).
3. In the left sidebar, click **Collaborators** under *Access*.
4. Click **Add people** (green button).
5. Enter your instructor's GitHub username or email and click **Add <username> to this repository**.
6. Ensure you **do not remove** the instructor during the evaluation and training period.

---

## 2. Step 9: GitHub Issues Setup
Under **Step 9**, create and track issues on GitHub representing core project milestones. Below are the recommended issues with pre-formatted titles and descriptions:

### Issue #1: Dataset Collection, Splitting & Data Augmentation Pipeline
- **Title:** `[Data] Dataset Collection, Splitting & Data Augmentation Pipeline`
- **Labels:** `data`, `enhancement`
- **Assignee:** `arnavarya11`
- **Body:**
  ```markdown
  ### Description
  Ingest the Stanford Cars dataset (`mteb/StanfordCars`) from Hugging Face Datasets and construct a high-throughput PyTorch DataLoader pipeline.

  ### Tasks
  - [x] Stream/download Stanford Cars dataset (16,185 total samples, 196 classes).
  - [x] Create 80/20 train/validation split (6,515 train, 1,629 validation, 8,041 test).
  - [x] Implement torchvision data transformations (`RandomHorizontalFlip`, `RandomRotation(10)`, ImageNet normalization).
  - [x] Implement `code/dataset.py` with custom batch collation.
  ```

### Issue #2: Vision Transformer (ViT-B/16) Model Implementation
- **Title:** `[Model] Vision Transformer (ViT-B/16) Architecture & Transfer Learning Adaptation`
- **Labels:** `model`, `architecture`
- **Assignee:** `arnavarya11`
- **Body:**
  ```markdown
  ### Description
  Instantiate and adapt the Vision Transformer (`google/vit-base-patch16-224`) for 196-class fine-grained vehicle recognition.

  ### Tasks
  - [x] Load pretrained ViT-B/16 checkpoint from Hugging Face Transformers.
  - [x] Adapt linear classification head from 1000 to 196 outputs with `ignore_mismatched_sizes=True`.
  - [x] Implement `StanfordCarsViT` module in `code/model.py`.
  - [x] Verify parameter counts and GPU memory footprint.
  ```

### Issue #3: Model Training Execution & Checkpoint Management
- **Title:** `[Training] End-to-End Training Loop with AdamW & Validation Checkpointing`
- **Labels:** `training`
- **Assignee:** `arnavarya11`
- **Body:**
  ```markdown
  ### Description
  Implement and execute the full training loop with AdamW optimizer, tracking per-epoch loss and accuracy.

  ### Tasks
  - [x] Implement AdamW optimizer ($\text{lr} = 2 \times 10^{-5}$, $\text{weight decay} = 10^{-4}$).
  - [x] Implement per-epoch evaluation against the validation set.
  - [x] Checkpoint the optimal model weights (`best_vit_stanford_cars.pth`) based on validation accuracy.
  - [x] Save training history metrics to `training_history.json`.
  ```

### Issue #4: Comprehensive Testing & Fine-Grained Error Diagnostics
- **Title:** `[Testing] Model Evaluation on Test Set, Metrics & Confusion Matrix`
- **Labels:** `testing`, `evaluation`
- **Assignee:** `arnavarya11`
- **Body:**
  ```markdown
  ### Description
  Evaluate the model on the 8,041-image test set and generate multi-class diagnostic reports.

  ### Tasks
  - [x] Measure Top-1 Accuracy (77.02%) and Top-5 Accuracy (94.80%).
  - [x] Generate precision, recall, weighted F1, and macro F1 scores.
  - [x] Plot and save 196x196 confusion matrix heatmap.
  - [x] Isolate top-10 best and top-10 lowest performing vehicle classes.
  ```

### Issue #5: Standalone CLI Inference Tool & Deployment
- **Title:** `[Deployment] CLI Inference Script for Arbitrary Vehicle Image Prediction`
- **Labels:** `deployment`, `cli`
- **Assignee:** `arnavarya11`
- **Body:**
  ```markdown
  ### Description
  Build a command-line inference script enabling users to pass an arbitrary vehicle image and receive top-5 predictions with confidence bars.

  ### Tasks
  - [x] Implement `code/predict.py` with argument parsing (`--image`, `--checkpoint`, `--top_k`).
  - [x] Add visualization rendering top-1 prediction and horizontal bar chart of candidate probabilities.
  - [x] Test inference on real-world test images (Audi R8, Mini Cooper).
  ```

### Issue #6: Capstone Documentation, Presentations & Submission Deliverables
- **Title:** `[Documentation] Capstone Documentation, Presentation & Final Deliverables`
- **Labels:** `documentation`
- **Assignee:** `arnavarya11`
- **Body:**
  ```markdown
  ### Description
  Compile all mandatory capstone documentation, presentation files, screenshots, results, and installation guides.

  ### Tasks
  - [x] Author comprehensive `README.md` and `INSTALLATION.md`.
  - [x] Document presentation slides in `presentations/presentation_overview.md`.
  - [x] Generate all CSV results summaries and JSON metric archives.
  - [x] Prepare formal technical project report.
  ```

---

## 3. Step 10: Pull Request (PR) Workflow
Under **Step 10**, show active use of branches and Pull Requests:

1. **Create a Feature Branch:**
   ```bash
   git checkout -b feature/model-evaluation
   ```
2. **Commit and Push Changes:**
   ```bash
   git add code/evaluate.py capstone/results/
   git commit -m "feat(eval): add comprehensive evaluation script and test results"
   git push origin feature/model-evaluation
   ```
3. **Open Pull Request on GitHub:**
   - Base branch: `main`
   - Compare branch: `feature/model-evaluation`
   - PR Title: `feat(eval): Implement 196-class evaluation and confusion matrix diagnostics`
   - Reference the Issue: `Closes #4`
4. **Conduct Review & Merge:**
   - Review changes on the PR "Files Changed" tab.
   - Click **Merge Pull Request** and **Confirm Merge**.

---

## 4. Step 7 & Step 12: Team Capstone Repository & Final Submission
- **Step 7:** If collaborating in a group, create the team capstone repository (e.g. `BatchF-Capstone-GroupX`) and add team members and your instructor as collaborators.
- **Step 12:** Submit the following URLs on your course submission portal:
  - **Personal Repository URL:** `https://github.com/arnavarya11/MUJ-DS-2430010362`
  - **Capstone Repository URL:** `https://github.com/arnavarya11/MUJ-DS-2430010362` (or group repository URL)
  - **Presentation File:** `presentations/Stanford_Cars_Image_Classification_Capstone.pptx`
  - **Documentation & Results:** `README.md`, `INSTALLATION.md`, `capstone/`
