"""
Dataset Loading and Preprocessing Pipeline for Stanford Cars
Author: Arnav Arya (Reg No: 2430010362)
"""

import torch
from torchvision import transforms
from datasets import load_dataset
from torch.utils.data import DataLoader


def get_data_transforms(image_size=(224, 224)):
    """
    Returns image transformation pipelines for training and validation/testing.
    Standard ImageNet normalization is used for Vision Transformer (ViT-B/16).
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    return train_transform, eval_transform


def load_stanford_cars_dataset(dataset_name="mteb/StanfordCars", val_split_ratio=0.2, seed=42):
    """
    Loads Stanford Cars dataset from Hugging Face and splits training set
    into train (80%) and validation (20%) partitions.

    Returns:
        tuple: (train_split, val_split, test_split, class_names)
    """
    raw_dataset = load_dataset(dataset_name)
    class_names = raw_dataset["train"].features["label"].names

    # Stratified or randomized 80/20 train/validation split
    train_val_split = raw_dataset["train"].train_test_split(test_size=val_split_ratio, seed=seed)
    train_split = train_val_split["train"]
    val_split = train_val_split["test"]
    test_split = raw_dataset["test"]

    train_transform, eval_transform = get_data_transforms()

    # Apply on-the-fly transforms
    train_split = train_split.with_transform(
        lambda examples: {
            "pixel_values": [train_transform(img.convert("RGB")) for img in examples["image"]],
            "label": examples["label"]
        }
    )

    val_split = val_split.with_transform(
        lambda examples: {
            "pixel_values": [eval_transform(img.convert("RGB")) for img in examples["image"]],
            "label": examples["label"]
        }
    )

    test_split = test_split.with_transform(
        lambda examples: {
            "pixel_values": [eval_transform(img.convert("RGB")) for img in examples["image"]],
            "label": examples["label"]
        }
    )

    return train_split, val_split, test_split, class_names


def collate_fn(batch):
    """
    Collate function to stack transformed tensors into PyTorch batches.
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return {"pixel_values": pixel_values, "labels": labels}


def create_dataloaders(train_data, val_data, test_data=None, batch_size=16, num_workers=2):
    """
    Creates PyTorch DataLoaders with custom collation.
    """
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers
    )

    test_loader = None
    if test_data is not None:
        test_loader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers
        )

    return train_loader, val_loader, test_loader
