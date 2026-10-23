"""
Vision Transformer (ViT) Model Architecture for Stanford Cars
Author: Arnav Arya (Reg No: 2430010362)
"""

import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig


class StanfordCarsViT(nn.Module):
    """
    Vision Transformer (ViT-B/16) tailored for fine-grained 196-class vehicle recognition.
    """
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        num_classes: int = 196,
        pretrained: bool = True,
        id2label: dict = None,
        label2id: dict = None,
        freeze_backbone: bool = False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.model_name = model_name

        if pretrained:
            self.vit = ViTForImageClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True
            )
        else:
            config = ViTConfig.from_pretrained(model_name, num_labels=num_classes)
            self.vit = ViTForImageClassification(config)

        if freeze_backbone:
            for param in self.vit.vit.parameters():
                param.requires_grad = False

    def forward(self, pixel_values, labels=None):
        return self.vit(pixel_values=pixel_values, labels=labels)

    def get_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {"trainable": trainable, "total": total, "percent_trainable": (trainable / total) * 100}


def build_vit_classifier(
    model_name="google/vit-base-patch16-224",
    num_classes=196,
    pretrained=True,
    class_names=None,
    freeze_backbone=False,
    device=None
):
    """
    Convenience factory function to build and initialize the ViT model.
    """
    id2label = None
    label2id = None
    if class_names:
        id2label = {i: name for i, name in enumerate(class_names)}
        label2id = {name: i for i, name in enumerate(class_names)}

    model = StanfordCarsViT(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        id2label=id2label,
        label2id=label2id,
        freeze_backbone=freeze_backbone
    )

    if device is not None:
        model = model.to(device)

    return model
