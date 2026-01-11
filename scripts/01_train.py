#!/usr/bin/env python
"""
scripts/01_train.py

CLI entrypoint for training (medimg_baseline_cls).

New in this version
1) Persists model architecture (arch) into the run's config.json (post-hoc update).
   - This avoids modifying src/config.py while still recording arch for reproducibility.
   - Downstream loaders should tolerate extra keys (02_eval_gradcam.py does).

Key features
- Robust project-root resolution (can run from any working directory)
- CLI overrides for dataset path and core hyperparameters
- Writes run artifacts under outputs/runs/<project>_<timestamp>/
- Writes outputs/runs/_latest_run.json pointer for downstream evaluation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple


# Resolve repo root: <repo>/scripts/01_train.py -> parents[1] == <repo>
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_image_size(s: str) -> Tuple[int, int]:
    s = s.lower().strip().replace("x", ",").replace(" ", ",")
    parts = [p for p in s.split(",") if p != ""]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Use --image-size like 224,224 or 224x224.")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as e:
        raise argparse.ArgumentTypeError("Image size must be two integers.") from e


def _update_run_config_json(run_dir: Path, updates: dict) -> None:
    """Update <run_dir>/config.json in-place with extra metadata (e.g., arch).

    This is intentionally post-hoc so we do not need to modify Config schema.
    """
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Expected config.json not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    changed = False
    for k, v in updates.items():
        if data.get(k) != v:
            data[k] = v
            changed = True

    if changed:
        with cfg_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Updated run config: {cfg_path} (added/updated keys: {list(updates.keys())})")


def main() -> None:
    from src.config import Config, seed_everything, ensure_dirs
    from src.utils import env_report, save_json
    from src.data import build_datasets, build_loaders, label_counts
    from src.models import build_model, freeze_backbone, get_head_prefixes, get_gradcam_target_layer
    from src.train import run_training

    parser = argparse.ArgumentParser(description="Train a 2D CXR classifier (medimg_baseline_cls).")
    parser.add_argument("--data-root", type=str, required=True, help="Dataset root with class subfolders (NORMAL/, PNEUMONIA/).")

    parser.add_argument("--project-name", type=str, default="medimg_baseline_cls", help="Project name used for run folder naming.")
    parser.add_argument("--output-root", type=str, default="outputs", help="Output root (contains runs/).")

    parser.add_argument("--arch", type=str, default="resnet18", help="Backbone architecture (must match your build_model support).")
    parser.add_argument("--image-size", type=_parse_image_size, default=(224, 224), help="Image size, e.g., 224,224.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true", default=True)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")

    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--head-epochs", type=int, default=2)
    parser.add_argument("--lr-head", type=float, default=3e-3)
    parser.add_argument("--lr-finetune", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--rebuild-balanced-val", action="store_true", default=True)
    parser.add_argument("--no-rebuild-balanced-val", dest="rebuild_balanced_val", action="store_false")
    parser.add_argument("--val-n-per-class", type=int, default=200)

    parser.add_argument("--use-weighted-sampler", action="store_true", default=True)
    parser.add_argument("--no-weighted-sampler", dest="use_weighted_sampler", action="store_false")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument("--device", type=str, default=None, help='Override device (e.g., "cuda" or "cpu").')

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

    if args.device is not None:
        cfg.device = args.device

    print("Project root:", ROOT)
    seed_everything(cfg.seed, cfg.deterministic)
    ensure_dirs(cfg)

    print("Config:", cfg)
    print("Environment:", env_report())

    ds = build_datasets(
        root_dir=cfg.data_root,
        class_names=cfg.class_names,
        image_size=cfg.image_size,
        rebuild_balanced_val=cfg.rebuild_balanced_val,
        val_n_per_class=cfg.val_n_per_class,
        seed=cfg.seed,
    )

    print("Counts:")
    print("train:", label_counts(ds["train_items"], len(cfg.class_names)))
    print("val  :", label_counts(ds["val_items"], len(cfg.class_names)))
    print("test :", label_counts(ds["test_items"], len(cfg.class_names)))

    loaders = build_loaders(
        train_ds=ds["train_ds"],
        val_ds=ds["val_ds"],
        test_ds=ds["test_ds"],
        train_items=ds["train_items"],
        class_names=cfg.class_names,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        use_weighted_sampler=cfg.use_weighted_sampler,
    )

    arch = args.arch
    model = build_model(arch=arch, num_classes=len(cfg.class_names), pretrained=True, device=cfg.device)

    freeze_backbone(model, head_prefixes=get_head_prefixes(arch))
    print("Trainable tensors:", sum(p.requires_grad for p in model.parameters()), "/", len(list(model.parameters())))

    target_layer = get_gradcam_target_layer(model, arch)
    print("Grad-CAM target layer:", target_layer)

    result = run_training(
        cfg=cfg,
        model=model,
        train_loader=loaders["train_loader"],
        val_loader=loaders["val_loader"],
        test_loader=loaders["test_loader"],
    )

    run_dir = Path(result["run_dir"]).resolve()
    print("Run saved to:", run_dir)
    print("Best checkpoint:", result["best_ckpt_path"])
    print("VAL summary:", result["val_summary"])
    print("TEST summary:", result["test_summary"])

    save_json(env_report(), str(run_dir / "env_report.json"))

    # Persist arch into config.json for this run
    _update_run_config_json(run_dir, {"arch": arch})

    latest_path = Path(cfg.output_root) / "runs" / "_latest_run.json"
    save_json(
        {"run_dir": str(run_dir), "best_ckpt_path": result["best_ckpt_path"], "arch": arch},
        str(latest_path),
    )
    print("Wrote latest run pointer to:", latest_path)


if __name__ == "__main__":
    main()
