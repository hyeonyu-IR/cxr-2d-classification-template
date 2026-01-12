#!/usr/bin/env python
"""
scripts/03_make_report.py

Generate a lightweight, self-contained run report from saved artifacts.

Inputs (expected in <run_dir>/):
- config.json
- history.csv (epoch-level training/val metrics)
- eval_summary.json (produced by 02_eval_gradcam.py)
- errors_<strategy>.csv (optional; produced by 02_eval_gradcam.py)
- gradcam/FP and gradcam/FN folders (optional)

Outputs (written to <run_dir>/<out-subdir>/):
- figures/*.png (learning curves, PR curve, confusion matrices)
- tables/* (copies of key run artifacts)
- index.html (single HTML report referencing relative assets)

This script is intentionally non-interactive and is designed to be run from a terminal.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _load_json(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def resolve_run_dir(run_arg: str, runs_root: Path) -> Path:
    """Resolve a run directory. Supports 'latest' via runs_root/_latest_run.json."""
    if run_arg.lower() == "latest":
        meta = runs_root / "_latest_run.json"
        if not meta.exists():
            raise FileNotFoundError(f"Latest-run pointer not found: {meta}")
        d = _load_json(meta)
        run_dir = Path(d["run_dir"])
        if not run_dir.is_absolute():
            run_dir = (Path.cwd() / run_dir).resolve()
        return run_dir.resolve()

    run_dir = Path(run_arg)
    if not run_dir.is_absolute():
        run_dir = (Path.cwd() / run_dir).resolve()
    return run_dir.resolve()


def _read_history(history_path: Path) -> Optional[pd.DataFrame]:
    if not history_path.exists():
        return None
    df = pd.read_csv(history_path)
    df.columns = [c.strip() for c in df.columns]
    return df


def _plot_learning_curves(df: pd.DataFrame, fig_dir: Path) -> None:
    """Plot common training curves if present.

    Expects some subset of columns:
      - epoch (optional; otherwise uses row index)
      - train_loss, val_loss
      - train_acc,  val_acc
      - train_ap,   val_ap

    Produces one PNG per metric found.
    """
    candidates = [
        ("train_loss", "val_loss", "Loss"),
        ("train_acc", "val_acc", "Accuracy"),
        ("train_ap", "val_ap", "Average Precision"),
    ]

    x = df["epoch"] if "epoch" in df.columns else (np.arange(len(df)) + 1)

    for a, b, title in candidates:
        if a not in df.columns and b not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
        if a in df.columns:
            ax.plot(x, df[a].values, label=a)
        if b in df.columns:
            ax.plot(x, df[b].values, label=b)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        metric_png = fig_dir / f"learning_{title.lower().replace(' ', '_')}.png"
        fig.savefig(metric_png, bbox_inches="tight")
        plt.close(fig)


def _plot_confusion_matrix(cm: np.ndarray, title: str, out_png: Path, class_names: Tuple[str, str]) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.0), dpi=160)
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _plot_pr_curve(y_true: np.ndarray, y_score: np.ndarray, out_png: Path) -> None:
    order = np.argsort(-y_score)
    y_true_s = y_true[order]

    tp = np.cumsum(y_true_s == 1)
    fp = np.cumsum(y_true_s == 0)

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp[-1], 1)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _copy_some_images(src_dir: Path, dst_dir: Path, max_images: int) -> List[str]:
    if not src_dir.exists():
        return []
    _safe_mkdir(dst_dir)
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    files = [p for p in sorted(src_dir.glob("*")) if p.suffix.lower() in exts]
    files = files[:max_images]
    copied: List[str] = []
    for p in files:
        out = dst_dir / p.name
        try:
            shutil.copy2(p, out)
            copied.append(out.name)
        except Exception:
            pass
    return copied


def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def build_html_report(
    out_html: Path,
    run_dir: Path,
    cfg: dict,
    eval_summary: Optional[dict],
    figures: Dict[str, str],
    tables: Dict[str, str],
    gradcam_thumbs: Dict[str, List[str]],
    title: str,
) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    class_names = cfg.get("class_names", ["NEG", "POS"])
    if isinstance(class_names, list) and len(class_names) >= 2:
        cn_text = ", ".join([str(class_names[0]), str(class_names[1])])
    else:
        cn_text = "NEG, POS"

    arch = cfg.get("arch", "(not recorded)")
    data_root = cfg.get("data_root", "")
    run_name = run_dir.name

    def img_tag(relpath: str, width: int = 900) -> str:
        return f'<div class="img"><img src="{relpath}" style="max-width:{width}px;width:100%;height:auto;"/></div>'

    metrics_block = ""
    if eval_summary is not None:
        val = eval_summary.get("val", {})
        test = eval_summary.get("test", {})
        metrics_block = f"""
        <h3>Evaluation summary</h3>
        <div class="kv">
          <div><b>Arch</b>: {_html_escape(str(eval_summary.get('arch', arch)))}</div>
          <div><b>Threshold strategy</b>: {_html_escape(str(eval_summary.get('threshold_strategy', '')))}</div>
          <div><b>Threshold used</b>: {_html_escape(str(eval_summary.get('threshold_used', '')))}</div>
        </div>

        <h4>Validation</h4>
        <ul>
          <li>Loss: {_html_escape(str(val.get('loss', '')))}</li>
          <li>Accuracy: {_html_escape(str(val.get('acc', '')))}</li>
          <li>Average Precision: {_html_escape(str(val.get('ap', '')))}</li>
          <li>thr_f1: {_html_escape(str(val.get('thr_f1', '')))}</li>
          <li>thr_sens: {_html_escape(str(val.get('thr_sens', '')))}</li>
        </ul>

        <h4>Test</h4>
        <ul>
          <li>Loss: {_html_escape(str(test.get('loss', '')))}</li>
          <li>Accuracy: {_html_escape(str(test.get('acc', '')))}</li>
          <li>Average Precision: {_html_escape(str(test.get('ap', '')))}</li>
        </ul>
        """

    figs_html = ""
    if figures:
        figs_html += "<h3>Figures</h3>"
        for name, rel in figures.items():
            figs_html += f"<h4>{_html_escape(name)}</h4>" + img_tag(rel)

    tables_html = ""
    if tables:
        tables_html += "<h3>Artifacts</h3><ul>"
        for name, rel in tables.items():
            tables_html += f'<li><a href="{rel}">{_html_escape(name)}</a></li>'
        tables_html += "</ul>"

    gradcam_html = ""
    if gradcam_thumbs:
        gradcam_html += "<h3>Grad-CAM samples (copied)</h3>"
        for split_name, imgs in gradcam_thumbs.items():
            if not imgs:
                continue
            gradcam_html += f"<h4>{_html_escape(split_name)}</h4><div class=\\"grid\\">"
            for fn in imgs:
                rel = f"gradcam/{split_name}/{fn}"
                gradcam_html += f'<a href="{rel}"><img src="{rel}" loading="lazy"/></a>'
            gradcam_html += "</div>"

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{_html_escape(title)} - {run_name}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; max-width: 1100px; }}
    code, pre {{ background: #f6f8fa; padding: 2px 4px; border-radius: 4px; }}
    pre {{ padding: 12px; overflow-x: auto; }}
    .kv div {{ margin: 4px 0; }}
    .img {{ margin: 12px 0 20px 0; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 10px;
    }}
    .grid img {{ width: 100%; height: auto; border: 1px solid #ddd; border-radius: 6px; }}
    .muted {{ color: #666; }}
  </style>
</head>
<body>
  <h1>{_html_escape(title)}</h1>
  <div class="muted">Generated: {ts}</div>

  <h2>Run</h2>
  <div class="kv">
    <div><b>Run directory</b>: {_html_escape(str(run_dir))}</div>
    <div><b>Run name</b>: {_html_escape(run_name)}</div>
    <div><b>Architecture</b>: {_html_escape(str(arch))}</div>
    <div><b>Classes</b>: {_html_escape(cn_text)}</div>
    <div><b>Data root</b>: {_html_escape(str(data_root))}</div>
  </div>

  {metrics_block}

  {figs_html}

  {tables_html}

  {gradcam_html}

  <hr/>
  <div class="muted">
    This report is generated from saved artifacts (history.csv, eval_summary.json, errors CSV, and Grad-CAM images).
    It is intended for research review and qualitative error analysis.
  </div>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a report folder (figures + HTML) for a run.")
    parser.add_argument("--run", type=str, default="latest", help='Run directory path, or "latest".')
    parser.add_argument("--runs-root", type=str, default=str(Path("outputs") / "runs"), help="Runs root containing _latest_run.json.")
    parser.add_argument("--out-subdir", type=str, default="report", help="Subfolder under run_dir to write the report.")
    parser.add_argument("--max-gradcam", type=int, default=12, help="Max images per Grad-CAM category to copy into report.")
    parser.add_argument("--copy-gradcam", action="store_true", default=True, help="Copy Grad-CAM samples into the report folder.")
    parser.add_argument("--no-copy-gradcam", dest="copy_gradcam", action="store_false")
    parser.add_argument("--title", type=str, default="Run report", help="Report title (used in HTML).")

    args = parser.parse_args()

    runs_root = Path(args.runs_root).expanduser().resolve()
    run_dir = resolve_run_dir(args.run, runs_root)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    out_dir = run_dir / args.out_subdir
    fig_dir = out_dir / "figures"
    tbl_dir = out_dir / "tables"
    _safe_mkdir(fig_dir)
    _safe_mkdir(tbl_dir)

    cfg_path = run_dir / "config.json"
    cfg = _load_json(cfg_path) if cfg_path.exists() else {}

    history_path = run_dir / "history.csv"
    hist = _read_history(history_path)

    eval_path = run_dir / "eval_summary.json"
    eval_summary = _load_json(eval_path) if eval_path.exists() else None

    figures: Dict[str, str] = {}
    tables: Dict[str, str] = {}
    gradcam_thumbs: Dict[str, List[str]] = {}

    # Learning curves
    if hist is not None:
        _plot_learning_curves(hist, fig_dir)
        for p in sorted(fig_dir.glob("learning_*.png")):
            figures[p.stem] = f"figures/{p.name}"
        try:
            shutil.copy2(history_path, tbl_dir / "history.csv")
            tables["history.csv"] = "tables/history.csv"
        except Exception:
            pass

    # Confusion matrices from eval_summary.json
    if eval_summary is not None:
        test = eval_summary.get("test", {})
        cm_f1 = test.get("cm_thr_f1")
        cm_sens = test.get("cm_thr_sens")

        class_names = cfg.get("class_names", ["NEG", "POS"])
        if isinstance(class_names, list) and len(class_names) >= 2:
            cn = (str(class_names[0]), str(class_names[1]))
        else:
            cn = ("NEG", "POS")

        if cm_f1 is not None:
            cm = np.array(cm_f1, dtype=int)
            out_png = fig_dir / "cm_test_maxf1.png"
            _plot_confusion_matrix(cm, "TEST confusion matrix (Max-F1 threshold)", out_png, cn)
            figures["cm_test_maxf1"] = f"figures/{out_png.name}"

        if cm_sens is not None:
            cm = np.array(cm_sens, dtype=int)
            out_png = fig_dir / "cm_test_targetsens.png"
            _plot_confusion_matrix(cm, "TEST confusion matrix (Target-sensitivity threshold)", out_png, cn)
            figures["cm_test_targetsens"] = f"figures/{out_png.name}"

        try:
            shutil.copy2(eval_path, tbl_dir / "eval_summary.json")
            tables["eval_summary.json"] = "tables/eval_summary.json"
        except Exception:
            pass

    # Optional PR curve if raw arrays exist (you can add these outputs later if desired)
    y_true_path = run_dir / "test_y_true.npy"
    y_score_path = run_dir / "test_y_score.npy"
    if y_true_path.exists() and y_score_path.exists():
        y_true = np.load(y_true_path).astype(int)
        y_score = np.load(y_score_path).astype(float)
        out_png = fig_dir / "pr_curve_test.png"
        _plot_pr_curve(y_true, y_score, out_png)
        figures["pr_curve_test"] = f"figures/{out_png.name}"
        try:
            shutil.copy2(y_true_path, tbl_dir / "test_y_true.npy")
            shutil.copy2(y_score_path, tbl_dir / "test_y_score.npy")
            tables["test_y_true.npy"] = "tables/test_y_true.npy"
            tables["test_y_score.npy"] = "tables/test_y_score.npy"
        except Exception:
            pass

    # Copy error CSVs
    for p in sorted(run_dir.glob("errors_*.csv")):
        try:
            shutil.copy2(p, tbl_dir / p.name)
            tables[p.name] = f"tables/{p.name}"
        except Exception:
            pass

    # Copy Grad-CAM thumbnails for portability
    if args.copy_gradcam:
        gc_root = run_dir / "gradcam"
        for split in ["FP", "FN"]:
            copied = _copy_some_images(gc_root / split, out_dir / "gradcam" / split, max_images=args.max_gradcam)
            gradcam_thumbs[split] = copied

    # Write HTML
    out_html = out_dir / "index.html"
    build_html_report(
        out_html=out_html,
        run_dir=run_dir,
        cfg=cfg,
        eval_summary=eval_summary,
        figures=figures,
        tables=tables,
        gradcam_thumbs=gradcam_thumbs,
        title=args.title,
    )

    print("Report written to:", out_html)
    print("Open in a browser (double-click):", out_html)


if __name__ == "__main__":
    main()
