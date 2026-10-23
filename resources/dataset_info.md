# Stanford Cars Dataset Information

## 1. Overview
The **Stanford Cars** dataset is a benchmark dataset for fine-grained visual categorization (FGVC), originally created by Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei at the Stanford AI Lab.

- **Total Images:** 16,185
- **Number of Classes:** 196 fine-grained categories
- **Splits in Repository:**
  - **Training Set (80%):** 6,515 images
  - **Validation Set (20%):** 1,629 images
  - **Test Set:** 8,041 images
- **Hugging Face Identifier:** `mteb/StanfordCars` (derived from the official benchmark)

## 2. Granularity and Taxonomy
The classes are organized at the level of **Make, Model, and Year**:
- Example 1: `AM General Hummer SUV 2000`
- Example 2: `Audi R8 Coupe 2012`
- Example 3: `Mini Cooper Roadster Convertible 2012`
- Example 4: `Tesla Model S Sedan 2012`

Each class represents subtle visual differences:
- Subtle grille variations between model years
- Wheel rim variations and trim packages
- Roof silhouettes (Coupe vs. Convertible vs. Sedan vs. Hatchback)
- Variations in lighting, viewpoint, background clutter, and vehicle occlusion

## 3. Data Preprocessing & Augmentation Pipeline
For Vision Transformer (ViT-B/16) input compatibility:
- **Spatial Resolution:** $224 \times 224$ pixels
- **Training Augmentation:**
  - `transforms.Resize((224, 224))`
  - `transforms.RandomHorizontalFlip(p=0.5)`
  - `transforms.RandomRotation(degrees=10)`
  - `transforms.ToTensor()`
  - `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`
- **Validation / Test Pipeline:**
  - `transforms.Resize((224, 224))`
  - `transforms.ToTensor()`
  - `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`

## 4. Visual Samples
Below is a sample batch visualization illustrating diverse camera angles, vehicle colors, and environmental settings across 8 car categories:

![Stanford Cars Samples](sample_data_grid.png)

## 5. Dataset Citations
```bibtex
@inproceedings{krause20133d,
  title={3D Object Representations for Fine-Grained Categorization},
  author={Krause, Jonathan and Stark, Michael and Deng, Jia and Fei-Fei, Li},
  booktitle={4th IEEE Workshop on 3D Representation and Recognition (3dRR-13)},
  year={2013}
}
```
