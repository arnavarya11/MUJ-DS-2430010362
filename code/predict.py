"""
Inference Script: Predict Vehicle Make, Model, and Year for Any Input Image
Author: Arnav Arya (Reg No: 2430010362)
"""

import os
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

from dataset import get_data_transforms
from model import build_vit_classifier
from utils import load_class_names


def predict_image(image_path, model, class_names, top_k=5, device="cpu"):
    """
    Runs inference on an input image, returning top-k predictions with confidence percentages.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    _, eval_transform = get_data_transforms()
    raw_img = Image.open(image_path).convert("RGB")
    tensor_img = eval_transform(raw_img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(pixel_values=tensor_img)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        probs = F.softmax(logits, dim=1)
        top_probs, top_indices = torch.topk(probs, k=top_k, dim=1)

    results = []
    for prob, idx in zip(top_probs[0].tolist(), top_indices[0].tolist()):
        class_name = class_names[idx] if idx < len(class_names) else f"Class {idx}"
        results.append({"class_name": class_name, "class_index": idx, "confidence": prob * 100})

    return results, raw_img


def visualize_prediction(image, predictions, save_path=None):
    """
    Renders the image alongside a horizontal bar chart of top-k predicted vehicle classes.
    """
    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(12, 5))

    ax_img.imshow(image)
    ax_img.axis("off")
    ax_img.set_title(f"Top-1: {predictions[0]['class_name']}", fontsize=12, fontweight="bold")

    names = [p["class_name"] for p in reversed(predictions)]
    scores = [p["confidence"] for p in reversed(predictions)]

    ax_bar.barh(names, scores, color="#2b5c8f")
    ax_bar.set_xlabel("Confidence (%)", fontsize=11)
    ax_bar.set_xlim(0, 100)
    for i, v in enumerate(scores):
        ax_bar.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] Prediction visualization saved to {save_path}")
    plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Predict vehicle make and model from image")
    parser.add_argument("--image", type=str, required=True, help="Path to input car image")
    parser.add_argument("--checkpoint", type=str, default="../checkpoints/best_vit_stanford_cars.pth", help="Model checkpoint path")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top predictions to display")
    parser.add_argument("--save_vis", type=str, default=None, help="Save path for visualization plot")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = load_class_names()

    model = build_vit_classifier(num_classes=len(class_names), class_names=class_names, device=device)
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"[INFO] Loaded checkpoint weights from {args.checkpoint}")
    else:
        print(f"[WARNING] Checkpoint {args.checkpoint} not found. Running with base model weights.")

    preds, img = predict_image(args.image, model, class_names, top_k=args.top_k, device=device)

    print("\n" + "=" * 50)
    print("Top-K Vehicle Predictions:")
    print("=" * 50)
    for i, p in enumerate(preds, 1):
        print(f"{i}. {p['class_name']} ({p['confidence']:.2f}%)")
    print("=" * 50 + "\n")

    if args.save_vis:
        visualize_prediction(img, preds, save_path=args.save_vis)


if __name__ == "__main__":
    main()
