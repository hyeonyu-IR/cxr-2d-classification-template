#!/usr/bin/env python
"""
02_eval_gradcam.py

CLI entrypoint for evaluating a trained checkpoint and producing Grad-CAM panels.

Design goals
- Portable: resolves project root from script location; can be run from any working directory.
- Reproducible: resolves latest run via outputs/runs/_latest_run.json or accepts an explicit run directory.
- Script-friendly: argparse + main().
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

# Resolve project root robustly (scripts/ -> repo root)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def resolve_run_dir(run_arg: str, runs_root: Path) -> Path:
    """
    Resolve run directory from:
      - "latest": use outputs/runs/_latest_run.json pointer
      - explicit path: outputs/runs/<run_name> or any valid path
    """
    if run_arg.lower() == "latest":
        meta_path = runs_root / "_latest_run.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Latest-run pointer not found: {meta_path}. "
                "Run training first, or pass --run <RUN_DIR>."
            )
        from src.utils import load_json
        meta = load_json(str(meta_path))
        if "run_dir" not in meta:
            raise KeyError(f"'run_dir' key not found in {meta_path}. Keys: {list(meta.keys())}")

        run_dir = Path(meta["run_dir"])
        if not run_dir.is_absolute():
            run_dir = (Path.cwd() / run_dir).resolve()
        return run_dir.resolve()

    run_dir = Path(run_arg)
    if not run_dir.is_absolute():
        run_dir = (Path.cwd() / run_dir).resolve()
    return run_dir.resolve()


def load_config_from_run(run_dir: Path):
    from src.utils import load_json
    from src.config import Config

    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found in run directory: {cfg_path}")

    cfg_dict = load_json(str(cfg_path))
    cfg = Config(**cfg_dict)
    cfg.run_dir = str(run_dir)  # optional convenience
    return cfg


def main() -> None:
    from src.config import seed_everything
    from src.utils import load_json
    from src.data import build_datasets, build_loaders
    from src.models import build_model, get_gradcam_target_layer
    from src.eval import eval_split, find_best_threshold, compute_confusion
    from src.gradcam import make_gradcam, save_gradcam_batch

    parser = argparse.ArgumentParser(description="Evaluate a trained run and generate Grad-CAM panels.")
    parser.add_argument(
        "--run",
        type=str,
        default="latest",
        help='Run directory path, or "latest" to use outputs/runs/_latest_run.json',
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default=str(Path("outputs") / "runs"),
        help="Root directory containing run folders and _latest_run.json",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Checkpoint filename inside the run directory (e.g., best.pt). If omitted, uses pointer best_ckpt_path if available.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Override cfg.device (e.g., "cuda" or "cpu"). If omitted, uses cfg.device.',
    )
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        help="Override architecture for model construction if not stored elsewhere. If omitted, uses the script's default or your run's expectation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of cases/images to process for Grad-CAM (debugging).",
    )
    parser.add_argument("--n-fp", type=int, default=10, help="Number of false positives to export as Grad-CAM panels.")
    parser.add_argument("--n-fn", type=int, default=10, help="Number of false negatives to export as Grad-CAM panels.")
    parser.add_argument(
        "--out-subdir",
        type=str,
        default="gradcam",
        help="Subdirectory under run_dir to write Grad-CAM outputs.",
    )

    args = parser.parse_args()

    runs_root = Path(args.runs_root).expanduser().resolve()
    run_dir = resolve_run_dir(args.run, runs_root)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    cfg = load_config_from_run(run_dir)

    # Device override
    if args.device is not None:
        cfg.device = args.device

    seed_everything(cfg.seed, cfg.deterministic)

    # Resolve checkpoint path
    best_ckpt_path: Optional[Path] = None

    # Prefer explicit --ckpt (relative to run_dir if not absolute)
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
        if not ckpt_path.is_absolute():
            ckpt_path = (run_dir / ckpt_path).resolve()
        best_ckpt_path = ckpt_path
    else:
        # Try to use latest pointer if available
        meta_path = runs_root / "_latest_run.json"
        if meta_path.exists():
            meta = load_json(str(meta_path))
            if "best_ckpt_path" in meta and meta["best_ckpt_path"]:
                ckpt_path = Path(meta["best_ckpt_path"])
                if not ckpt_path.is_absolute():
                    ckpt_path = (Path.cwd() / ckpt_path).resolve()
                best_ckpt_path = ckpt_path

    if best_ckpt_path is None or not best_ckpt_path.exists():
        # Fallback: common names inside run_dir
        candidates = [
            run_dir / "best.pt",
            run_dir / "best.pth",
            run_dir / "model_best.pt",
            run_dir / "checkpoints" / "best.pt",
        ]
        best_ckpt_path = next((p for p in candidates if p.exists()), None)

    if best_ckpt_path is None or not best_ckpt_path.exists():
        raise FileNotFoundError(
            "Could not resolve checkpoint. Provide --ckpt, or ensure _latest_run.json has best_ckpt_path, "
            "or place a best checkpoint under the run directory."
        )

    print("Project root:", ROOT)
    print("Using run_dir:", run_dir)
    print("Using checkpoint:", best_ckpt_path)
    print(cfg)

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
        use_weighted_sampler=False,  # not needed for eval
    )

    val_loader = loaders["val_loader"]
    test_loader = loaders["test_loader"]

    # Build model
    # NOTE: your training script likely fixed arch (e.g., resnet18). If you store arch in config later,
    # you can remove --arch override and load it from cfg.
    arch = args.arch or "resnet18"
    model = build_model(arch=arch, num_classes=len(cfg.class_names), pretrained=False, device=cfg.device)

    # Load weights
    import torch
    state = torch.load(str(best_ckpt_path), map_location=cfg.device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()

    # Criterion for eval (if required by your eval functions)
    import torch.nn as nn
    criterion = nn.CrossEntropyLoss()

    # Evaluate
    val_out = eval_split(
        model=model, loader=val_loader, device=cfg.device, criterion=criterion,
        class_names=cfg.class_names, pos_class_name=cfg.pos_class_name
    )
    test_out = eval_split(
        model=model, loader=test_loader, device=cfg.device, criterion=criterion,
        class_names=cfg.class_names, pos_class_name=cfg.pos_class_name
    )

    print("=== Checkpoint performance ===")
    print(f"VAL  | loss={val_out['loss']:.4f} acc={val_out['acc']:.4f} AP={val_out.get('ap', float('nan')):.4f}")
    print(f"TEST | loss={test_out['loss']:.4f} acc={test_out['acc']:.4f} AP={test_out.get('ap', float('nan')):.4f}")

    # Thresholds from VAL
    best_thr = find_best_threshold(val_out["y_true"], val_out["y_prob"])
    print("Best threshold (VAL):", best_thr)

    # Confusion on TEST using best_thr
    cm = compute_confusion(test_out["y_true"], test_out["y_prob"], thr=best_thr)
    print("Confusion matrix (TEST):")
    print(cm)

    # Grad-CAM setup
    target_layer = get_gradcam_target_layer(model, arch)
    gradcam = make_gradcam(model=model, target_layer=target_layer, alpha=cfg.gradcam_alpha)

    # Identify FP/FN indices (assuming pos class is encoded as 1)
    y_true = test_out["y_true"]
    y_prob = test_out["y_prob"]
    y_pred = (y_prob >= best_thr).astype(int)

    fp_idx = [i for i, (yt, yp) in enumerate(zip(y_true, y_pred)) if yt == 0 and yp == 1]
    fn_idx = [i for i, (yt, yp) in enumerate(zip(y_true, y_pred)) if yt == 1 and yp == 0]

    # Limit for debugging (applies to FP/FN selection)
    n_fp = min(args.n_fp, len(fp_idx))
    n_fn = min(args.n_fn, len(fn_idx))

    # Extract file paths for selected items. Your eval_split/test_out must provide "items" or similar.
    # In your original notebook-derived script, this was sourced from datasets or cached item dicts.
    # We'll reproduce the prior behavior: datasets["test_items"] is expected to be available.
    test_items = datasets.get("test_items", None)
    if test_items is None:
        # Common fallback: dataset object may have .items
        test_ds = datasets["test_ds"]
        test_items = getattr(test_ds, "items", None)

    if test_items is None:
        raise RuntimeError(
            "Cannot find test_items to locate image file paths for Grad-CAM export. "
            "Ensure build_datasets returns 'test_items' or that test_ds has an '.items' attribute."
        )

    fp_paths = [test_items[i]["image"] for i in fp_idx[:n_fp]]
    fn_paths = [test_items[i]["image"] for i in fn_idx[:n_fn]]

    out_dir = Path(run_dir) / args.out_subdir
    (out_dir / "FP").mkdir(parents=True, exist_ok=True)
    (out_dir / "FN").mkdir(parents=True, exist_ok=True)

    records_fp = save_gradcam_batch(
        model=model,
        gradcam=gradcam,
        image_paths=fp_paths,
        out_dir=str(out_dir / "FP"),
        class_names=cfg.class_names,
        device=cfg.device,
        image_size=cfg.image_size,
        limit=args.limit,
    ) if "limit" in save_gradcam_batch.__code__.co_varnames else save_gradcam_batch(
        model, gradcam, fp_paths, str(out_dir / "FP"), cfg.class_names, cfg.device, cfg.image_size
    )

    records_fn = save_gradcam_batch(
        model=model,
        gradcam=gradcam,
        image_paths=fn_paths,
        out_dir=str(out_dir / "FN"),
        class_names=cfg.class_names,
        device=cfg.device,
        image_size=cfg.image_size,
        limit=args.limit,
    ) if "limit" in save_gradcam_batch.__code__.co_varnames else save_gradcam_batch(
        model, gradcam, fn_paths, str(out_dir / "FN"), cfg.class_names, cfg.device, cfg.image_size
    )

    print("Saved FP Grad-CAM panels:", len(records_fp))
    print("Saved FN Grad-CAM panels:", len(records_fn))
    print("Grad-CAM output directory:", out_dir)


if __name__ == "__main__":
    main()
