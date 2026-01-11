#!/usr/bin/env python
"""
01_train.py

CLI entrypoint for training a 2D chest X-ray classifier using the medimg_baseline_cls template.

Design goals
- Reproducible: config is saved into a run directory; a "_latest_run.json" pointer is written for downstream evaluation.
- Portable: no machine-specific hard-coded paths; project root is resolved from this script location.
- Script-friendly: all execution is behind a main() entrypoint with argparse.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

# Resolve project root robustly (scripts/ -> repo root)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_image_size(s: str) -> Tuple[int, int]:
    """
    Parse image size from:
      - "224,224"
      - "224x224"
      - "224 224"
    """
    s = s.lower().replace("x", ",").replace(" ", ",")
    parts = [p for p in s.split(",") if p.strip() != ""]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Invalid --image-size '{s}'. Use e.g. 224,224 or 224x224.")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid --image-size '{s}'. Must be two integers.") from e


def main() -> None:
    from src.config import Config, seed_everything, ensure_dirs
    from src.utils import env_report, save_json
    from src.data import build_datasets, build_loaders
    from src.models import build_model, freeze_backbone, get_head_prefixes
    from src.train import run_training

    parser = argparse.ArgumentParser(description="Train a 2D classification model (medimg_baseline_cls).")
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Dataset root directory containing class subfolders (e.g., NORMAL/, PNEUMONIA/).",
    )
    parser.add_argument("--project-name", type=str, default="medimg_baseline_cls", help="Project name for run folder naming.")
    parser.add_argument("--output-root", type=str, default="outputs", help="Root folder for outputs (runs, artifacts).")

    # Model + training
    parser.add_argument("--arch", type=str, default="resnet18", help="Backbone architecture (e.g., resnet18, densenet121).")
    parser.add_argument("--image-size", type=_parse_image_size, default=(224, 224), help="Image size, e.g. 224,224.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (start 0 on Windows; increase when stable).")
    parser.add_argument("--pin-memory", action="store_true", help="Enable DataLoader pin_memory.")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false", help="Disable DataLoader pin_memory.")
    parser.set_defaults(pin_memory=True)

    parser.add_argument("--max-epochs", type=int, default=10, help="Total epochs.")
    parser.add_argument("--head-epochs", type=int, default=2, help="Head-only epochs before fine-tuning.")
    parser.add_argument("--lr-head", type=float, default=3e-3, help="Learning rate for head-only stage.")
    parser.add_argument("--lr-finetune", type=float, default=1e-3, help="Learning rate for fine-tuning stage.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")

    # Validation balancing
    parser.add_argument("--rebuild-balanced-val", action="store_true", help="Rebuild balanced validation set.")
    parser.add_argument("--no-rebuild-balanced-val", dest="rebuild_balanced_val", action="store_false", help="Do not rebuild balanced validation set.")
    parser.set_defaults(rebuild_balanced_val=True)

    parser.add_argument("--val-n-per-class", type=int, default=200, help="Number of items per class for balanced val/test sampling.")
    parser.add_argument("--use-weighted-sampler", action="store_true", help="Use weighted sampler for training.")
    parser.add_argument("--no-weighted-sampler", dest="use_weighted_sampler", action="store_false", help="Disable weighted sampler.")
    parser.set_defaults(use_weighted_sampler=True)

    # Repro + device
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic behavior (may reduce speed).")
    parser.add_argument("--no-deterministic", dest="deterministic", action="store_false", help="Disable deterministic behavior.")
    parser.set_defaults(deterministic=False)

    parser.add_argument("--device", type=str, default=None, help='Override device (e.g., "cuda" or "cpu"). Default uses Config default.')

    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"--data-root does not exist: {data_root}")

    cfg = Config(
        project_name=args.project_name,
        data_root=str(data_root),
        output_root=args.output_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        max_epochs=args.max_epochs,
        head_epochs=args.head_epochs,
        lr_head=args.lr_head,
        lr_finetune=args.lr_finetune,
        weight_decay=args.weight_decay,
        rebuild_balanced_val=args.rebuild_balanced_val,
        val_n_per_class=args.val_n_per_class,
        use_weighted_sampler=args.use_weighted_sampler,
        seed=args.seed,
        deterministic=args.deterministic,
    )

    # Optional device override
    if args.device is not None:
        cfg.device = args.device

    seed_everything(cfg.seed, cfg.deterministic)
    ensure_dirs(cfg)

    print("Project root:", ROOT)
    print("Config:", cfg)
    print("Environment:", env_report())

    # Data
    datasets = build_datasets(
        root_dir=cfg.data_root,
        class_names=cfg.class_names,
        image_size=cfg.image_size,
        rebuild_balanced_val=cfg.rebuild_balanced_val,
        val_n_per_class=cfg.val_n_per_class,
    )

    loaders = build_loaders(
        train_ds=datasets["train_ds"],
        val_ds=datasets["val_ds"],
        test_ds=datasets["test_ds"],
        class_names=cfg.class_names,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        use_weighted_sampler=cfg.use_weighted_sampler,
    )

    # Model
    model = build_model(
        arch=args.arch,
        num_classes=len(cfg.class_names),
        pretrained=True,
        device=cfg.device,
    )

    # Freeze backbone for head-only stage
    freeze_backbone(model, head_prefixes=get_head_prefixes(args.arch))

    # Train
    result = run_training(
        cfg=cfg,
        model=model,
        train_loader=loaders["train_loader"],
        val_loader=loaders["val_loader"],
        test_loader=loaders["test_loader"],
        arch=args.arch,
    ) if "arch" in run_training.__code__.co_varnames else run_training(
        cfg=cfg,
        model=model,
        train_loader=loaders["train_loader"],
        val_loader=loaders["val_loader"],
        test_loader=loaders["test_loader"],
    )

    run_dir = Path(result["run_dir"]).resolve()
    print("Run saved to:", run_dir)
    print("Best checkpoint:", result.get("best_ckpt_path"))
    print("VAL summary:", result.get("val_summary"))
    print("TEST summary:", result.get("test_summary"))

    # Save environment report into the run directory for reproducibility
    save_json(env_report(), str(run_dir / "env_report.json"))

    # Save a pointer to the latest run for downstream scripts/notebooks
    runs_root = Path(cfg.output_root) / "runs"
    latest_path = runs_root / "_latest_run.json"
    save_json(
        {"run_dir": str(run_dir), "best_ckpt_path": result.get("best_ckpt_path")},
        str(latest_path),
    )
    print("Wrote latest run pointer to:", latest_path)


if __name__ == "__main__":
    main()
