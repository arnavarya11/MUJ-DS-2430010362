"""
Training Script for Vision Transformer on Stanford Cars Dataset
Author: Arnav Arya (Reg No: 2430010362)
"""

import os
import time
import json
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataset import load_stanford_cars_dataset, create_dataloaders
from model import build_vit_classifier
from utils import plot_training_history


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in train_loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * pixel_values.size(0)
        _, preds = torch.max(logits, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            loss = criterion(logits, labels)
            running_loss += loss.item() * pixel_values.size(0)
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def main():
    parser = argparse.ArgumentParser(description="Train ViT-B/16 on Stanford Cars")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for AdamW")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--model_name", type=str, default="google/vit-base-patch16-224", help="Base model")
    parser.add_argument("--output_dir", type=str, default="../capstone/results", help="Directory for output results")
    parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints", help="Directory to save weights")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    print("[INFO] Loading Stanford Cars dataset...")
    train_split, val_split, test_split, class_names = load_stanford_cars_dataset()
    train_loader, val_loader, test_loader = create_dataloaders(
        train_split, val_split, test_split, batch_size=args.batch_size
    )

    print(f"[INFO] Classes: {len(class_names)} | Train: {len(train_split)} | Val: {len(val_split)}")

    model = build_vit_classifier(
        model_name=args.model_name,
        num_classes=len(class_names),
        pretrained=True,
        class_names=class_names,
        device=device
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    best_checkpoint_path = os.path.join(args.checkpoint_dir, "best_vit_stanford_cars.pth")

    history = {
        "epochs": list(range(1, args.epochs + 1)),
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "best_val_accuracy": 0.0
    }

    print("\n[INFO] Starting training loop...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        elapsed = (time.time() - t0) / 60

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        is_best = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            history["best_val_accuracy"] = best_val_acc
            torch.save(model.state_dict(), best_checkpoint_path)
            is_best = "⭐ BEST MODEL SAVED"

        print(
            f"Epoch [{epoch}/{args.epochs}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}% | "
            f"Time: {elapsed:.1f} min {is_best}"
        )

    # Save history and curves
    with open(os.path.join(args.output_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

    plot_training_history(history, save_dir=args.output_dir)
    print(f"\n[INFO] Training complete! Best validation accuracy: {best_val_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
