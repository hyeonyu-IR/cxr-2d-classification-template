# Script Usage Guide

This document describes how to run the **script-based training and evaluation workflows**
provided in this repository. These scripts are intended for **reproducible, non-interactive**
experiments and are preferred over notebooks for repeated runs.

---

## Environment

Activate the conda environment before running any scripts:

```bash
conda activate medimg
```
## Training (scripts/01_train.py)
### Basic usage (required argument)
```
python scripts/01_train.py --data-root "/path/to/chest_xray"
```
```--data-root``` must point to a directory with class subfolders (e.g., NORMAL/, PNEUMONIA/).

### Common optional overrides
```
python scripts/01_train.py \
  --data-root "/path/to/chest_xray" \
  --arch resnet18 \
  --batch-size 32 \
  --max-epochs 10 \
  --head-epochs 2
```

### Outputs
Each run creates a timestamped directory:
```
outputs/runs/<project>_<timestamp>/
```

### Key artifacts:
- ```best.pt``` – best checkpoint (selected by validation AP)
- ```history.csv``` – epoch-level metrics
- ```config.json``` – experiment configuration (includes arch)
- ```env_report.json``` – environment metadata

### A pointer file is also written:
```
outputs/runs/_latest_run.json
```
This enables downstream evaluation with ```--run latest```.
---

## Evaluation and Grad-CAM (scripts/02_eval_gradcam.py)
### Evaluate the most recent run
```
python scripts/02_eval_gradcam.py --run latest
```
### Evaluate a specific run
```
python scripts/02_eval_gradcam.py \
  --run outputs/runs/medimg_baseline_cls_YYYYMMDD_HHMMSS
```
The script:
- reloads the best checkpoint,
- evaluates validation and test splits,
- selects thresholds on the validation set,
- reports test-set metrics and confusion matrices,
- identifies hard false positives / false negatives,
- exports Grad-CAM visualizations.

---

## Threshold strategies
Two threshold selection strategies are computed on the validation set:

1) Max-F1 (default)
```
python scripts/02_eval_gradcam.py --run latest --threshold-strategy maxf1
```
- Uses the threshold that maximizes F1 score on validation
- Often yields very high sensitivity but may increase false positives

2) Target sensitivity
```
python scripts/02_eval_gradcam.py \
  --run latest \
  --threshold-strategy targetsens \
  --target-sens 0.95
```
- Chooses the lowest threshold achieving sensitivity ≥ target on validation
- Useful when prioritizing sensitivity while reducing false positives

The selected strategy controls:
- hard-error mining
- Grad-CAM export
- error CSV generation

---

## Grad-CAM outputs
Saved under the run directory:
```
<run_dir>/gradcam/
├── FP/   # false positives
└── FN/   # false negatives
```
Each image includes:
- original image
- Grad-CAM heatmap
- overlay with predicted probability

---

## Error CSV export
For each evaluation run, a CSV file is written:
```
<run_dir>/errors_<strategy>.csv
```
Example:
- errors_maxf1.csv
- errors_targetsens.csv

Each row includes:
- sample index
- error type (FP / FN)
- threshold used
- image path
- true label
- predicted score
- predicted label
These files are suitable for Excel, R, or downstream analysis.

---

## Help
All scripts expose detailed CLI help:
```
python scripts/01_train.py --help
python scripts/02_eval_gradcam.py --help
```