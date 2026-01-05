# src/config.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple
import json
import random
import numpy as np
import torch


@dataclass
class Config:
    # --------------------
    # Project / paths
    # --------------------
    project_name: str = "medimg_baseline_cls"
    data_root: str = ""  # set in notebook or script
    output_root: str = "outputs"

    # --------------------
    # Dataset
    # --------------------
    class_names: Tuple[str, ...] = ("NORMAL", "PNEUMONIA")
    pos_class_name: str = "PNEUMONIA"
    image_size: Tuple[int, int] = (224, 224)

    # Validation rebuild
    rebuild_balanced_val: bool = True
    val_n_per_class: int = 200

    # --------------------
    # Dataloader
    # --------------------
    batch_size: int = 32
    num_workers: int = 0          # Windows-safe default; increase cautiously
    pin_memory: bool = True

    # --------------------
    # Training
    # --------------------
    max_epochs: int = 10
    head_epochs: int = 2

    lr_head: float = 3e-3
    lr_finetune: float = 1e-3     # consider 3e-4 for stability
    weight_decay: float = 1e-4

    use_weighted_sampler: bool = True

    # --------------------
    # Reproducibility
    # --------------------
    seed: int = 42
    deterministic: bool = False   # set True only if strict determinism is required

    # --------------------
    # Device
    # --------------------
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------
    # Logging / checkpoints
    # --------------------
    save_best_by: str = "val_ap"  # criterion to select best checkpoint
    save_history: bool = True

    # --------------------
    # Interpretability
    # --------------------
    gradcam_alpha: float = 0.35


def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    """
    Seed Python, NumPy, and PyTorch for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def ensure_dirs(cfg: Config) -> None:
    """
    Create output directories if they do not exist.
    """
    Path(cfg.output_root).mkdir(parents=True, exist_ok=True)
    (Path(cfg.output_root) / "runs").mkdir(parents=True, exist_ok=True)
    (Path(cfg.output_root) / "gradcam").mkdir(parents=True, exist_ok=True)


def save_config(cfg: Config, out_path: str) -> None:
    """
    Save config to JSON for reproducibility.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(asdict(cfg), f, indent=2)
