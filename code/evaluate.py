"""
Evaluation Script for Vision Transformer on Stanford Cars Dataset
Author: Arnav Arya (Reg No: 2430010362)
"""

import os
import json
import csv
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import load_stanford_cars_dataset, create_dataloaders
from model import build_vit_classifier
from utils import calculate_metrics, plot_confusion_matrix, load_class_names


def evaluate_model(model, test_loader, device):
    """
    Evaluates the model over the test set, returning predictions, probabilities, and ground truth labels.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            probs = F.softmax(logits, dim=1)

            _, preds = torch.max(logits, 1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    return all_labels, all_preds, all_probs


def main():
    parser = argparse.ArgumentParser(description="Evaluate ViT-B/16 on Stanford Cars")
    parser.add_argument("--checkpoint", type=str, default="../checkpoints/best_vit_stanford_cars.pth", help="Path to checkpoint")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--output_dir", type=str, default="../capstone/results", help="Directory to save evaluation results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[INFO] Loading dataset...")
    train_split, val_split, test_split, class_names = load_stanford_cars_dataset()
    _, _, test_loader = create_dataloaders(
        train_split, val_split, test_split, batch_size=args.batch_size
    )

    model = build_vit_classifier(num_classes=len(class_names), class_names=class_names, device=device)

    if os.path.exists(args.checkpoint):
        print(f"[INFO] Loading checkpoint from {args.checkpoint}...")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    else:
        print(f"[WARNING] Checkpoint not found at {args.checkpoint}. Evaluating with initialized weights or simulated metrics.")

    y_true, y_pred, y_probs = evaluate_model(model, test_loader, device)
    metrics = calculate_metrics(y_true, y_pred, y_probs, top_k=5)

    print("\n================ Evaluation Results ================")
    print(f"Top-1 Accuracy:     {metrics['top_1_accuracy'] * 100:.2f}%")
    print(f"Top-5 Accuracy:     {metrics['top_5_accuracy'] * 100:.2f}%")
    print(f"Weighted Precision: {metrics['weighted_precision'] * 100:.2f}%")
    print(f"Weighted Recall:    {metrics['weighted_recall'] * 100:.2f}%")
    print(f"Weighted F1-Score:  {metrics['weighted_f1'] * 100:.2f}%")
    print(f"Macro Precision:    {metrics['macro_precision'] * 100:.2f}%")
    print(f"Macro Recall:       {metrics['macro_recall'] * 100:.2f}%")
    print(f"Macro F1-Score:     {metrics['macro_f1'] * 100:.2f}%")
    print("====================================================\n")

    # Save metrics
    metrics_path = os.path.join(args.output_dir, "final_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    # Plot confusion matrix
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, save_path=cm_path)
    print(f"[INFO] Confusion matrix saved to {cm_path}")


if __name__ == "__main__":
    main()
