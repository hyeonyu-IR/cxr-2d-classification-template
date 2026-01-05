# src/utils.py
from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def timestamp() -> str:
    """Return a sortable timestamp string."""
    return time.strftime("%Y%m%d_%H%M%S")


def make_run_dir(output_root: str, project_name: str, run_name: Optional[str] = None) -> Path:
    """
    Create and return a run directory:
      outputs/runs/<project>_<timestamp>[_<run_name>]
    """
    out = Path(output_root) / "runs"
    out.mkdir(parents=True, exist_ok=True)

    tag = f"{project_name}_{timestamp()}"
    if run_name:
        tag += f"_{run_name}"
    run_dir = out / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(obj: Any, path: str, indent: int = 2) -> None:
    """Save a Python object to JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if is_dataclass(obj):
        obj = asdict(obj)

    with open(p, "w") as f:
        json.dump(obj, f, indent=indent)


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def save_text(text: str, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)


def env_report() -> Dict[str, Any]:
    """
    Lightweight environment report for reproducibility.
    Good to save alongside outputs.
    """
    report = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        report.update(
            {
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
            }
        )
    return report


def recommended_num_workers() -> int:
    """
    Conservative guidance for Windows.
    Start small; increase if stable and IO-bound.
    """
    # On Windows, too many workers can cause instability; keep conservative.
    cpu = os.cpu_count() or 4
    if cpu <= 4:
        return 0
    if cpu <= 8:
        return 2
    return 4


def format_metrics(metrics: Dict[str, Any], keys=None, ndigits: int = 4) -> str:
    """
    Pretty-format numeric metrics.
    """
    if keys is None:
        keys = list(metrics.keys())

    parts = []
    for k in keys:
        if k not in metrics:
            continue
        v = metrics[k]
        if isinstance(v, float):
            parts.append(f"{k}={v:.{ndigits}f}")
        else:
            parts.append(f"{k}={v}")
    return " | ".join(parts)
