# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [Unreleased]

### Added
- Latest-run resolution for evaluation workflows using `outputs/runs/_latest_run.json` as a run pointer.
- Script-friendly evaluation entrypoint for Grad-CAM evaluation (see `scripts/02_eval_gradcam.py` or equivalent).

### Changed
- Grad-CAM evaluation workflow now loads configuration from `config.json` within the resolved run directory (instead of hard-coding a run folder name).
- Evaluation now decouples run-state metadata (e.g., run directory pointers) from the strict `Config` dataclass schema.

### Fixed
- Prevented schema mismatch errors when instantiating `Config` from JSON containing run-state keys (e.g., `run_dir`).

## [0.2.1] - 2026-01-10
### Changed
- Grad-CAM evaluation uses latest-run pointer for automatic run selection.
- Fixed: robust Grad-CAM image handling in manuscript document generator
