# Report Generation

This document describes the post-hoc report generation workflow provided by
`scripts/03_make_report.py`.

The report script is designed to convert saved experiment artifacts into a
self-contained, human-readable summary suitable for qualitative review,
error analysis, and manuscript preparation.

---

## Purpose

Training and evaluation scripts in this repository are intentionally
non-interactive and optimized for reproducibility. Visualization and
interpretation are handled **after** model execution by generating
persistent artifacts.

`scripts/03_make_report.py` aggregates these artifacts into a structured
report containing:
- performance summaries,
- publication-quality plots,
- links to tabular results,
- representative Grad-CAM examples.

---

## Expected inputs

The script operates on an existing run directory, typically located under:

## outputs/runs/<run_id>/

The following files may be used if present:

- `config.json` – experiment configuration (including model architecture)
- `history.csv` – epoch-level training and validation metrics
- `eval_summary.json` – evaluation metrics and confusion matrices
- `errors_<strategy>.csv` – structured false-positive / false-negative listings
- `gradcam/FP/`, `gradcam/FN/` – Grad-CAM visualization outputs

Missing files are handled gracefully; only available artifacts are included.

---

## Basic usage

Generate a report for the most recent run:

```bash
python scripts/03_make_report.py --run latest
python scripts/03_make_report.py \
  --run outputs/runs/medimg_baseline_cls_YYYYMMDD_HHMMSS
```

## Output structure
The report is written to a subdirectory within the run folder:
```
<run_dir>/report/
├── index.html
├── figures/
│   ├── learning_loss.png
│   ├── learning_accuracy.png
│   ├── cm_test_maxf1.png
│   └── cm_test_targetsens.png
├── tables/
│   ├── history.csv
│   ├── eval_summary.json
│   └── errors_<strategy>.csv
└── gradcam/
    ├── FP/
    └── FN/
```
The report can be viewed by opening ```index.html``` in any modern web browser.

---

## Contents of the report
### Performance summary
- Model architecture
- Threshold strategy used
- Validation and test metrics (loss, accuracy, average precision)

### Plots
- Training and validation learning curves (if available)
- Test-set confusion matrices at:
    - Max-F1 threshold
    - Target-sensitivity threshold

### Error analysis
- Links to CSV files containing false positives and false negatives
- Optional embedded Grad-CAM thumbnails for qualitative review

---

## Design philosophy
The report script follows these principles:
- Non-interactive: no GUI or Jupyter dependency
- Reproducible: derived entirely from saved artifacts
- Portable: a single folder that can be shared or archived
- Manuscript-friendly: figures and tables are ready for export

This separation allows training and evaluation to remain stable while
visualization and interpretation evolve independently.

---
## Typical workflow
1. Train a model:
   ```
   python scripts/01_train.py ...
   ```
2. Evaluate and generate Grad-CAMs:
   ```
   python scripts/02_eval_gradcam.py ...
   ```
3. Generate a report:
   ```
   python scripts/03_make_report.py --run latest
   ```
---
## Notes
- The report script does not modify model outputs.
- All generated figures are derived from saved metrics and predictions.
- The report is intended for research review and qualitative interpretation
and should not be used for clinical decision-making.







