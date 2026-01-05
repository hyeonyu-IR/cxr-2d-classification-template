# medimg_baseline_cls

A reproducible, modular baseline framework for **2D medical image classification** using PyTorch and MONAI, with built-in support for:

- pretrained CNN backbones (ResNet-18, DenseNet-121),
- class-imbalanced training,
- balanced validation construction,
- rigorous evaluation (AP, sensitivity/specificity, threshold selection),
- and **Grad-CAM–based visual interpretability**.

This repository was developed as a clean research template and can be reused for other radiology classification tasks (CXR, radiographs, pathology slides, etc.).

---
```
## Project structure

medimg_baseline_cls/
├── README.md
├── src/
│ ├── init.py
│ ├── config.py # Experiment configuration and seeding
│ ├── data.py # Dataset indexing, transforms, loaders
│ ├── models.py # Model factory + Grad-CAM target layers
│ ├── train.py # Head-only + fine-tuning training loop
│ ├── eval.py # Evaluation, metrics, thresholds, error mining
│ ├── interpret.py # Grad-CAM visualization and saving
│ └── utils.py # Misc utilities (logging, env report, helpers)
│
├── notebooks/
│ ├── 01_train.ipynb # Thin training driver
│ └── 02_eval_gradcam.ipynb # Evaluation + Grad-CAM driver
│
└── outputs/
└── runs/
└── <project>_<timestamp>/
├── best.pt
├── history.csv
├── config.json
├── env_report.json
├── val_summary.json
├── test_summary.json
├── eval_summary.json
└── gradcam/
├── FP/
└── FN/

## Dataset assumptions

This project currently assumes the **Kaggle Chest X-ray (Pneumonia)** directory structure:

chest_xray/
├── train/
│ ├── NORMAL/
│ └── PNEUMONIA/
├── test/
│ ├── NORMAL/
│ └── PNEUMONIA/
└── val/ # optional; can be rebuilt from train

```
Images may be grayscale or RGB.  
Grayscale images are automatically converted to **3-channel** format.

---

## Environment setup (example)

```bash
conda create -n medimg python=3.10
conda activate medimg

pip install torch torchvision torchaudio
pip install monai
pip install numpy pandas scikit-learn matplotlib tqdm pillow
```

GPU (CUDA) is automatically detected if available.

Typical workflow
1. Train a model

Open:
notebooks/01_train.ipynb

This notebook:
- builds datasets and loaders,
- trains a pretrained CNN in two stages:
    - Stage A: head-only training,
    - Stage B: full fine-tuning,
- selects the best checkpoint by validation average precision (AP),
- saves all artifacts under outputs/runs/<run_id>/.

Key outputs:
- best.pt – best model checkpoint
- history.csv – epoch-by-epoch metrics
- config.json – full experiment configuration
- env_report.json – environment metadata

2. Evaluate and generate Grad-CAMs

Open:
notebooks/02_eval_gradcam.ipynb

This notebook:
- reloads the best checkpoint,
- evaluates VAL and TEST splits,
- selects decision thresholds on VAL (Max-F1 and target sensitivity),
- reports TEST confusion matrices and metrics,
- identifies hard false positives / false negatives,
- generates and saves Grad-CAM visual explanations.
```
Grad-CAM outputs are saved to:
outputs/runs/<run_id>/gradcam/
├── FP/
└── FN/
```
Each image contains:
- original image,
- Grad-CAM heatmap,
- overlay with predicted probability.

Configuration
All experiment settings are defined in src/config.py via a single Config dataclass, including:
- data paths
- class names
- image size
- batch size
- learning rates
- number of epochs
- validation strategy
- device selection

This design ensures:
- reproducibility,
- clean experiment tracking,
- easy extension to new datasets.

Model backbones
Supported out of the box:
- resnet18 (default)
- densenet121

Switch backbones by changing a single line in the notebook driver:
arch = "resnet18"
Grad-CAM target layers are automatically selected per architecture.

Interpretability notes
Grad-CAM is intended to:
- verify that the model attends to anatomically plausible regions,
- identify shortcut learning (e.g., borders, text, markers),
- support qualitative error analysis.
Grad-CAM should not be interpreted as causal explanation.

Extending this template
Common next steps:
- add data augmentation (in data.py),
- add additional backbones (EfficientNet, ConvNeXt),
- adapt for multi-class classification,
- adapt for 2.5D or 3D inputs,
- integrate experiment tracking (e.g., MLflow).
The modular structure is designed to support these changes cleanly.

