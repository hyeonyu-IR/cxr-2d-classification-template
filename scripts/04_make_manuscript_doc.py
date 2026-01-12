#!/usr/bin/env python
"""
scripts/04_make_manuscript_doc.py

Generate a manuscript-ready Word document summarizing a completed run.
"""

from pathlib import Path
import argparse
import json
import pandas as pd
from docx import Document
from docx.shared import Inches
from datetime import datetime


def load_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_run_dir(run: str, runs_root: Path) -> Path:
    if run == "latest":
        meta = load_json(runs_root / "_latest_run.json")
        return Path(meta["run_dir"]).resolve()
    return Path(run).resolve()


def add_dataframe(doc: Document, df: pd.DataFrame, title: str):
    doc.add_heading(title, level=2)
    table = doc.add_table(rows=1, cols=len(df.columns))
    hdr_cells = table.rows[0].cells
    for i, col in enumerate(df.columns):
        hdr_cells[i].text = col

    for _, row in df.iterrows():
        row_cells = table.add_row().cells
        for i, val in enumerate(row):
            row_cells[i].text = str(val)


def add_images_from_folder(doc: Document, folder: Path, title: str):
    if not folder.exists():
        return

    doc.add_heading(title, level=2)

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp"}

    # Only include files (not directories like .ipynb_checkpoints)
    images = [p for p in sorted(folder.iterdir())
              if p.is_file() and p.suffix.lower() in exts]

    if not images:
        doc.add_paragraph("(No image files found.)")
        return

    for img in images:
        doc.add_paragraph(img.name)
        try:
            doc.add_picture(str(img), width=Inches(6))
        except Exception as e:
            # Do not fail the entire document build because of one file
            doc.add_paragraph(f"[Skipped: could not embed image: {img.name} | {type(e).__name__}: {e}]")
        doc.add_page_break()



def main():
    parser = argparse.ArgumentParser(description="Generate manuscript-ready Word document.")
    parser.add_argument("--run", default="latest", help="Run directory or 'latest'")
    parser.add_argument("--runs-root", default="outputs/runs")
    parser.add_argument("--out-name", default="manuscript_report.docx")
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run, Path(args.runs_root))
    out_path = run_dir / args.out_name

    doc = Document()
    doc.add_heading("Model Training and Evaluation Report", 0)
    doc.add_paragraph(f"Run directory: {run_dir}")
    doc.add_paragraph(f"Generated: {datetime.now().isoformat()}")

    # Environment
    env_path = run_dir / "env_report.json"
    if env_path.exists():
        doc.add_heading("Environment", level=1)
        env = load_json(env_path)
        for k, v in env.items():
            doc.add_paragraph(f"{k}: {v}")

    # Config
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        doc.add_heading("Configuration", level=1)
        cfg = load_json(cfg_path)
        for k, v in cfg.items():
            doc.add_paragraph(f"{k}: {v}")

    # History
    hist_path = run_dir / "history.csv"
    if hist_path.exists():
        df = pd.read_csv(hist_path)
        add_dataframe(doc, df, "Training History")

    # Evaluation summary
    eval_path = run_dir / "eval_summary.json"
    if eval_path.exists():
        doc.add_heading("Evaluation Summary", level=1)
        eval_summary = load_json(eval_path)
        for k, v in eval_summary.items():
            doc.add_paragraph(f"{k}: {v}")

    # Error tables
    for err_csv in run_dir.glob("errors_*.csv"):
        df = pd.read_csv(err_csv)
        add_dataframe(doc, df, f"Errors: {err_csv.name}")

    # Grad-CAM images
    add_images_from_folder(doc, run_dir / "gradcam" / "FP", "Grad-CAM: False Positives")
    add_images_from_folder(doc, run_dir / "gradcam" / "FN", "Grad-CAM: False Negatives")

    doc.save(out_path)
    print(f"Manuscript document written to: {out_path}")


if __name__ == "__main__":
    main()
