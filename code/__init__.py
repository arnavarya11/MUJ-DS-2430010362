"""
Stanford Cars Vision Transformer (ViT-B/16) Package
Author: Arnav Arya (Reg No: 2430010362)
Batch: Batch F - Capstone Project
"""

from .model import build_vit_classifier
from .dataset import load_stanford_cars_dataset, get_data_transforms, create_dataloaders
from .utils import calculate_metrics, plot_training_history, plot_confusion_matrix

__version__ = "1.0.0"
__author__ = "Arnav Arya"
