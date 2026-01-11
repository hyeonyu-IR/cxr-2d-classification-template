#!/usr/bin/env python
"""
scripts/02_eval_gradcam.py

CLI entrypoint for:
- Loading the most recent (or specified) run directory
- Rebuilding datasets/loaders
- Evaluating VAL/TEST and selecting thresholds on VAL
- Finding hard FP/FN errors
- Exporting Grad-CAM panels under <run_dir>/gradcam/

Aligned with your existing codebase:
- src.eval: eval_split, metrics_at_threshold, pick_threshold_max_f1, pick_threshold_target_sens, find_errors_binary
- src.interpret: GradCAM, save_gradcam_batch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def resolve_run_dir(run_arg: str, runs_root: Path) -> Path:
    if run_arg.lower() == "latest":
        meta_path = runs_root / "_latest_run.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Latest-run pointer not found: {meta_path}")
        from src.utils import load_json
        meta = load_json(str(meta_path))
        run_dir = Path(meta["run_dir"])
        if not run_dir.is_absolute():
            run_dir = (Path.cwd() / run_dir).resolve()
        return run_dir.resolve()

    run_dir = Path(run_arg)
    if not run_dir.is_absolute():
        run_dir = (Path.cwd() / run_dir).resolve()
    return run_dir.resolve()


def resolve_ckpt_path(run_dir: Path, runs_root: Path, ckpt_arg: Optional[str]) -> Path:
    from src.utils import load_json

    if ckpt_arg:
        p = Path(ckpt_arg)
        if not p.is_absolute():
            p = (run_dir / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"--ckpt not found: {p}")
        return p

    meta_path = runs_root / "_latest_run.json"
    if meta_path.exists():
        meta = load_json(str(meta_path))
        best = meta.get("best_ckpt_path")
        if best:
            p = Path(best)
            if not p.is_absolute():
                p = (Path.cwd() / p).resolve()
            if p.exists():
                return p

    p = (run_dir / "best.pt").resolve()
    if p.exists():
        return p

    raise FileNotFoundError("Could not resolve checkpoint. Provide --ckpt or ensure best.pt exists in the run directory.")


def main() -> None:
    from src.utils import load_json, save_json
    from src.config import Config, seed_everything
    from src.data import build_datasets, build_loaders
    from src.models import build_model, get_gradcam_target_layer
    from src.eval import (
        eval_split,
        metrics_at_threshold,
        pick_threshold_max_f1,
        pick_threshold_target_sens,
        find_errors_binary,
    )
    from src.interpret import GradCAM, save_gradcam_batch

    import torch
    import torch.nn as nn

    parser = argparse.ArgumentParser(description="Evaluate a run and export Grad-CAM panels.")
    parser.add_argument("--run", type=str, default="latest", help='Run directory path, or "latest".')
    parser.add_argument("--runs-root", type=str, default=str(Path("outputs") / "runs"), help="Runs root containing _latest_run.json.")
    parser.add_argument("--arch", type=str, default="resnet18", help="Architecture used during training (must match checkpoint).")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint filename/path. If relative, interpreted under run_dir.")
    parser.add_argument("--device", type=str, default=None, help='Override cfg.device (e.g., "cuda" or "cpu").')

    parser.add_argument("--target-sens", type=float, default=0.95, help="Target sensitivity for threshold selection on VAL.")
    parser.add_argument("--topk", type=int, default=10, help="Top-K hard errors to export (FP and FN indices).")
    parser.add_argument("--alpha", type=float, default=None, help="Override cfg.gradcam_alpha.")
    parser.add_argument("--out-subdir", type=str, default="gradcam", help="Output subdirectory under run_dir for Grad-CAM panels.")
    parser.add_argument("--save-summary", action="store_true", default=True)
    parser.add_argument("--no-save-summary", dest="save_summary", action="store_false")

    args = parser.parse_args()

    runs_root = Path(args.runs_root).expanduser().resolve()
    run_dir = resolve_run_dir(args.run, runs_root)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    cfg_dict = load_json(str(run_dir / "config.json"))
    cfg = Config(**cfg_dict)
    cfg.run_dir = str(run_dir)

    if args.device is not None:
        cfg.device = args.device
    if args.alpha is not None:
        cfg.gradcam_alpha = float(args.alpha)

    seed_everything(cfg.seed, cfg.deterministic)

    ckpt_path = resolve_ckpt_path(run_dir, runs_root, args.ckpt)

    print("Project root:", ROOT)
    print("Using run_dir:", run_dir)
    print("Using checkpoint:", ckpt_path)
    print(cfg)

    ds = build_datasets(
        root_dir=cfg.data_root,
        class_names=cfg.class_names,
        image_size=cfg.image_size,
        rebuild_balanced_val=cfg.rebuild_balanced_val,
        val_n_per_class=cfg.val_n_per_class,
        seed=cfg.seed,
    )

    loaders = build_loaders(
        train_ds=ds["train_ds"],
        val_ds=ds["val_ds"],
        test_ds=ds["test_ds"],
        train_items=ds["train_items"],
        class_names=cfg.class_names,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        use_weighted_sampler=False,
    )

    val_loader = loaders["val_loader"]
    test_loader = loaders["test_loader"]
    test_items = ds["test_items"]

    arch = args.arch
    model = build_model(arch=arch, num_classes=len(cfg.class_names), pretrained=False, device=cfg.device)

    state = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(state)
    model.to(cfg.device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    target_layer = get_gradcam_target_layer(model, arch)
    print("Grad-CAM target layer:", target_layer)

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

    thr_f1, best_f1 = pick_threshold_max_f1(val_out["y_true"], val_out["y_score"])
    m_val_f1 = metrics_at_threshold(val_out["y_true"], val_out["y_score"], thr_f1)

    thr_sens, m_val_sens = pick_threshold_target_sens(val_out["y_true"], val_out["y_score"], target_sens=args.target_sens)

    print("\n=== Thresholds selected on VAL ===")
    print(f"Max-F1 threshold: {thr_f1:.4f} (F1={best_f1:.3f})")
    print("VAL metrics @ thr_f1:", {k: v for k, v in m_val_f1.items() if k != "confusion_matrix"})
    print(f"\nTarget-sensitivity threshold: {thr_sens:.4f} (target sens >= {args.target_sens})")
    print("VAL metrics @ thr_sens:", {k: v for k, v in m_val_sens.items() if k != "confusion_matrix"})

    m_test_f1 = metrics_at_threshold(test_out["y_true"], test_out["y_score"], thr_f1)
    m_test_sens = metrics_at_threshold(test_out["y_true"], test_out["y_score"], thr_sens)

    print("\n=== TEST metrics using thresholds chosen on VAL ===")
    print(f"\n[TEST @ Max-F1 thr={thr_f1:.4f}]")
    print({k: v for k, v in m_test_f1.items() if k != "confusion_matrix"})
    print(f"\n[TEST @ TargetSens thr={thr_sens:.4f} (target sens {args.target_sens})]")
    print({k: v for k, v in m_test_sens.items() if k != "confusion_matrix"})
    print("\nConfusion matrix format: [[TN, FP],[FN, TP]]")
    print("\nTEST confusion @ thr_f1:\n", m_test_f1["confusion_matrix"])
    print("\nTEST confusion @ thr_sens:\n", m_test_sens["confusion_matrix"])

    errs = find_errors_binary(
        items=test_items,
        y_score=test_out["y_score"],
        pos_idx=test_out["pos_idx"],
        thr=thr_f1,
        topk=args.topk,
    )
    fp_idx = errs["fp_idx"]
    fn_idx = errs["fn_idx"]

    print("\n=== Hard errors on TEST (Max-F1 threshold) ===")
    print("Top false positives:")
    for i in fp_idx:
        print(f"  score={test_out['y_score'][i]:.3f} | path={test_items[i]['image']}")
    print("\nTop false negatives:")
    for i in fn_idx:
        print(f"  score={test_out['y_score'][i]:.3f} | path={test_items[i]['image']}")

    gradcam = GradCAM(model, target_layer)
    out_dir = Path(run_dir) / args.out_subdir
    fp_paths = [test_items[i]["image"] for i in fp_idx]
    fn_paths = [test_items[i]["image"] for i in fn_idx]

    records_fp = save_gradcam_batch(
        model=model,
        gradcam=gradcam,
        paths=fp_paths,
        out_dir=str(out_dir / "FP"),
        class_names=cfg.class_names,
        device=cfg.device,
        image_size=cfg.image_size,
        alpha=cfg.gradcam_alpha,
    )
    records_fn = save_gradcam_batch(
        model=model,
        gradcam=gradcam,
        paths=fn_paths,
        out_dir=str(out_dir / "FN"),
        class_names=cfg.class_names,
        device=cfg.device,
        image_size=cfg.image_size,
        alpha=cfg.gradcam_alpha,
    )

    print("Saved FP panels:", len(records_fp), "to", out_dir / "FP")
    print("Saved FN panels:", len(records_fn), "to", out_dir / "FN")

    if args.save_summary:
        summary = {
            "val": {
                "loss": float(val_out["loss"]),
                "acc": float(val_out["acc"]),
                "ap": float(val_out.get("ap", float("nan"))),
                "thr_f1": float(thr_f1),
                "thr_sens": float(thr_sens),
            },
            "test": {
                "loss": float(test_out["loss"]),
                "acc": float(test_out["acc"]),
                "ap": float(test_out.get("ap", float("nan"))),
                "metrics_at_thr_f1": {k: v for k, v in m_test_f1.items() if k != "confusion_matrix"},
                "metrics_at_thr_sens": {k: v for k, v in m_test_sens.items() if k != "confusion_matrix"},
                "cm_thr_f1": m_test_f1["confusion_matrix"].tolist(),
                "cm_thr_sens": m_test_sens["confusion_matrix"].tolist(),
            },
        }
        save_json(summary, str(Path(run_dir) / "eval_summary.json"))
        print("Saved:", Path(run_dir) / "eval_summary.json")


if __name__ == "__main__":
    main()
