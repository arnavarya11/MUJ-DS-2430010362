# Installation & Setup Guide

**Project Title:** Fine-Grained Vehicle Classification Using Vision Transformers (ViT-B/16)  
**Author:** Arnav Arya | Registration Number: 2430010362  
**Academic Batch:** Batch F – Capstone Project  
**Repository:** [https://github.com/arnavarya11/MUJ-DS-2430010362](https://github.com/arnavarya11/MUJ-DS-2430010362)  

---

## 1. System Requirements & Prerequisites
Ensure your development environment meets the following specifications before beginning:
- **Operating System:** Linux, macOS, or Windows 10/11
- **Python Version:** Python 3.10 or higher
- **Git:** Installed and configured
- **Hardware (Recommended):** NVIDIA GPU with at least 8 GB VRAM and CUDA 11.8+ for accelerated training and evaluation. CPU execution is supported for inference.

---

## 2. Local Environment Setup

### Step 2.1: Clone the Repository
Open a terminal (or PowerShell on Windows) and clone the repository:
```bash
git clone https://github.com/arnavarya11/MUJ-DS-2430010362.git
cd MUJ-DS-2430010362
```

### Step 2.2: Create and Activate a Virtual Environment
It is recommended to use an isolated Python virtual environment:

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Using Conda (Alternative):**
```bash
conda create -n stanford_cars python=3.10 -y
conda activate stanford_cars
```

### Step 2.3: Install Dependencies
Install all required libraries specified in `requirements.txt`:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Running with Google Colab (Zero-Setup GPU)
If local GPU resources are unavailable, the project can be run directly on Google Colab:

1. Navigate to [Google Colab](https://colab.research.google.com/).
2. Click **File** → **Upload notebook** and upload [`notebooks/2430010362.ipynb`](notebooks/2430010362.ipynb).
3. Connect a GPU runtime:
   - Click **Runtime** → **Change runtime type**.
   - Set **Hardware accelerator** to **T4 GPU** (or A100 if available).
   - Click **Save**.
4. Run all cells sequentially (**Runtime** → **Run all**).

---

## 4. Running the Modular Python Package

### Step 4.1: Training the Vision Transformer
To train the model from scratch using the CLI training script:
```bash
cd code
python train.py --epochs 10 --batch_size 16 --lr 2e-5 --weight_decay 1e-4 --output_dir ../capstone/results --checkpoint_dir ../checkpoints
```

**Available CLI Flags:**
- `--epochs`: Total training epochs (default: `10`).
- `--batch_size`: Batch size per step (default: `16`).
- `--lr`: Learning rate for AdamW optimizer (default: `2e-5`).
- `--weight_decay`: Weight decay factor (default: `1e-4`).
- `--output_dir`: Path to save training histories and performance curves (default: `../capstone/results`).
- `--checkpoint_dir`: Path to save optimal model weights (default: `../checkpoints`).

### Step 4.2: Evaluating Model Checkpoints
To evaluate model performance across the 8,041-image test set:
```bash
cd code
python evaluate.py --checkpoint ../checkpoints/best_vit_stanford_cars.pth --batch_size 16 --output_dir ../capstone/results
```

This generates:
- Top-1 & Top-5 accuracy figures
- Weighted and Macro Precision, Recall, and F1 scores
- `capstone/results/final_metrics.json`
- `capstone/results/confusion_matrix.png`

### Step 4.3: Running Inference on an Arbitrary Car Image
To predict the vehicle make, model, and year for any external image:
```bash
cd code
python predict.py --image path/to/vehicle.jpg --checkpoint ../checkpoints/best_vit_stanford_cars.pth --top_k 5 --save_vis ../capstone/screenshots/prediction_output.png
```

**Sample Output:**
```
==================================================
Top-K Vehicle Predictions:
==================================================
1. Audi R8 Coupe 2012 (69.81%)
2. Audi TT Hatchback 2011 (3.68%)
3. Audi TT RS Coupe 2012 (2.47%)
4. Audi S5 Convertible 2012 (2.33%)
5. Audi TTS Coupe 2012 (2.27%)
==================================================
```

---

## 5. Troubleshooting & FAQ

### Issue: CUDA Out of Memory (OOM)
- **Cause:** Batch size of 16 may exceed memory on GPUs with $\le 6\text{ GB}$ VRAM.
- **Solution:** Reduce the batch size using the `--batch_size 8` flag.

### Issue: Hugging Face Dataset Download Timeout
- **Cause:** Network latency when streaming `mteb/StanfordCars`.
- **Solution:** Enable Hugging Face offline cache or rerun with an established internet connection.

### Issue: Missing Classes File
- **Solution:** Ensure `resources/classes.json` exists in the repository root or run `code/utils.py` to regenerate the class taxonomy.
