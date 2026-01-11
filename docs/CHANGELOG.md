# Changelog

All notable changes to this repository are documented here.
The format loosely follows Keep a Changelog, with emphasis on research reproducibility.

---

## Unreleased (2026-01-11)

### Added
- Script-based training and evaluation workflows (`scripts/01_train.py`, `scripts/02_eval_gradcam.py`)
- Automatic persistence of model architecture (`arch`) into each run’s `config.json`
- `_latest_run.json` now records both run directory and architecture
- Threshold selection strategy flag (`--threshold-strategy {maxf1,targetsens}`)
- CSV export of hard false positives / false negatives for structured error analysis

### Changed
- Evaluation scripts now support safer checkpoint loading when available (`weights_only=True`)
- Grad-CAM and error mining are driven by the selected threshold strategy

### Fixed
- Loader invocation now consistently passes required dataset metadata (e.g., `train_items`)
