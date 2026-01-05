# src/data.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

from monai.data import Dataset
from monai.transforms import (
    Compose,
    LoadImageD,
    EnsureChannelFirstD,
    ScaleIntensityD,
    ResizeD,
    EnsureTypeD,
    Lambdad,
)

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler


def label_counts(items: List[Dict], n_classes: int) -> Dict[int, int]:
    counts = {i: 0 for i in range(n_classes)}
    for it in items:
        counts[int(it["label"])] += 1
    return counts


def build_kaggle_items(
    root_dir: str,
    split: str,
    class_names: Tuple[str, ...] = ("NORMAL", "PNEUMONIA"),
    exts: Tuple[str, ...] = (".jpeg", ".jpg", ".png"),
) -> List[Dict]:
    """
    Build a list of items from Kaggle chest_xray folder structure:
      root_dir/split/{NORMAL,PNEUMONIA}/*.jpeg

    Returns list of dicts: {"image": <path>, "label": <int>}
    where label index matches class_names order.
    """
    root = Path(root_dir)
    items: List[Dict] = []
    for label, cname in enumerate(class_names):
        d = root / split / cname
        if not d.exists():
            raise FileNotFoundError(f"Expected folder not found: {d}")
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.suffix.lower() in exts:
                items.append({"image": str(p), "label": int(label)})
    return items


def rebuild_val_balanced_from_train(
    train_items: List[Dict],
    n_per_class: int = 200,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Create a balanced validation set by sampling n_per_class from the training set
    for each class and removing them from training.

    Returns (new_train_items, new_val_items).
    """
    rng = np.random.default_rng(seed)

    labels = np.array([it["label"] for it in train_items], dtype=int)
    classes = sorted(np.unique(labels).tolist())
    val_indices = []

    for c in classes:
        idx = np.where(labels == c)[0]
        if len(idx) < n_per_class:
            raise ValueError(
                f"Not enough samples in class {c}: have {len(idx)}, need {n_per_class}."
            )
        chosen = rng.choice(idx, size=n_per_class, replace=False)
        val_indices.extend(chosen.tolist())

    val_set = set(val_indices)
    new_val = [train_items[i] for i in val_indices]
    new_train = [it for i, it in enumerate(train_items) if i not in val_set]

    return new_train, new_val


def to_3ch(x):
    """
    Convert (1,H,W) -> (3,H,W) by repeating; if already >=3 channels, keep first 3.
    Assumes EnsureChannelFirstD has been applied.
    """
    if x.ndim != 3:
        return x
    c = x.shape[0]
    if c == 1:
        return x.repeat(3, 1, 1)
    if c >= 3:
        return x[:3]
    return x


def build_transforms(
    image_size: Tuple[int, int] = (224, 224),
):
    """
    Phase-1 stable transforms: load -> channel-first -> 3ch -> scale [0,1] -> resize.
    No augmentation here; add augmentation later in a separate function.
    """
    tfm = Compose(
        [
            LoadImageD(keys=["image"], image_only=True),
            EnsureChannelFirstD(keys=["image"]),
            Lambdad(keys=["image"], func=to_3ch),
            ScaleIntensityD(keys=["image"]),
            ResizeD(keys=["image"], spatial_size=image_size),
            EnsureTypeD(keys=["image", "label"]),
        ]
    )
    return tfm


def build_datasets(
    root_dir: str,
    class_names: Tuple[str, ...] = ("NORMAL", "PNEUMONIA"),
    image_size: Tuple[int, int] = (224, 224),
    rebuild_balanced_val: bool = True,
    val_n_per_class: int = 200,
    seed: int = 42,
) -> Dict[str, object]:
    """
    Build train/val/test items and MONAI Datasets.
    If rebuild_balanced_val=True, sample val_n_per_class from train and create balanced val.
    """
    train_items = build_kaggle_items(root_dir, "train", class_names=class_names)
    test_items = build_kaggle_items(root_dir, "test", class_names=class_names)

    if rebuild_balanced_val:
        train_items, val_items = rebuild_val_balanced_from_train(
            train_items, n_per_class=val_n_per_class, seed=seed
        )
    else:
        val_items = build_kaggle_items(root_dir, "val", class_names=class_names)

    tfm = build_transforms(image_size=image_size)

    train_ds = Dataset(data=train_items, transform=tfm)
    val_ds = Dataset(data=val_items, transform=tfm)
    test_ds = Dataset(data=test_items, transform=tfm)

    return {
        "train_items": train_items,
        "val_items": val_items,
        "test_items": test_items,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
    }


def build_loaders(
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    train_items: List[Dict],
    class_names: Tuple[str, ...],
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
    use_weighted_sampler: bool = True,
) -> Dict[str, DataLoader]:
    """
    Build DataLoaders. For training, use WeightedRandomSampler by default
    to mitigate class imbalance.
    """
    if use_weighted_sampler:
        counts = label_counts(train_items, len(class_names))
        class_w = {c: 1.0 / counts[c] for c in counts}
        sample_weights = [class_w[it["label"]] for it in train_items]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }
