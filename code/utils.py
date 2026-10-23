"""
Utility Functions: Evaluation Metrics, Visualizations, and Logging
Author: Arnav Arya (Reg No: 2430010362)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def load_class_names(json_path=None):
    """
    Loads Stanford Cars 196 class names from resources/classes.json.
    """
    if json_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        json_path = os.path.join(base_dir, "resources", "classes.json")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("classes", data)


def calculate_metrics(y_true, y_pred, y_probs=None, top_k=5):
    """
    Calculates Top-1 accuracy, Top-5 accuracy, weighted and macro precision, recall, and F1.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    top1 = accuracy_score(y_true, y_pred)
    weighted_prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    top5 = None
    if y_probs is not None:
        y_probs = np.array(y_probs)
        top5_preds = np.argsort(y_probs, axis=1)[:, -top_k:]
        top5 = np.mean([y_true[i] in top5_preds[i] for i in range(len(y_true))])

    return {
        "top_1_accuracy": float(top1),
        "top_5_accuracy": float(top5) if top5 is not None else None,
        "weighted_precision": float(weighted_prec),
        "weighted_recall": float(weighted_rec),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "macro_f1": float(macro_f1)
    }


def plot_training_history(history, save_dir=None):
    """
    Plots training and validation loss and accuracy curves.
    """
    epochs = history.get("epochs", range(1, len(history["train_loss"]) + 1))

    # Loss plot
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, history["train_loss"], marker="o", color="#1f77b4", label="Training Loss", linewidth=2)
    ax.plot(epochs, history["val_loss"], marker="s", color="#ff7f0e", label="Validation Loss", linewidth=2)
    ax.set_title("Training vs Validation Loss", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Cross Entropy Loss", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    if save_dir:
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_loss_curve.png"), dpi=300)
    plt.close(fig)

    # Accuracy plot
    fig, ax = plt.subplots(figsize=(9, 5))
    train_acc = [acc * 100 if acc <= 1.0 else acc for acc in history["train_accuracy"]]
    val_acc = [acc * 100 if acc <= 1.0 else acc for acc in history["val_accuracy"]]
    ax.plot(epochs, train_acc, marker="o", color="#2ca02c", label="Training Accuracy", linewidth=2)
    ax.plot(epochs, val_acc, marker="s", color="#d62728", label="Validation Accuracy", linewidth=2)
    ax.set_title("Training vs Validation Accuracy", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    if save_dir:
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_accuracy_curve.png"), dpi=300)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plots and saves confusion matrix heatmap for 196 classes.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Confusion Matrix (196 Vehicle Classes)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Predicted Label Index", fontsize=12)
    ax.set_ylabel("True Label Index", fontsize=12)
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
    plt.close(fig)
