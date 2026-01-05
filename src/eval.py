# src/eval.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import sklearn.metrics as skm


@torch.no_grad()
def eval_split(
    model: torch.nn.Module,
    loader,
    device: str,
    criterion: Optional[torch.nn.Module] = None,
    class_names: Tuple[str, ...] = ("NORMAL", "PNEUMONIA"),
    pos_class_name: str = "PNEUMONIA",
) -> Dict:
    """
    Evaluate a model on a dataloader and return:
      - loss, acc
      - probs, labels, preds (argmax)
      - for binary classification: y_true (0/1), y_score (pos probability), ap

    Notes:
      - device is passed in explicitly to avoid hidden globals.
      - criterion is optional; if None, loss will be NaN.
    """
    model.eval()

    all_probs, all_labels = [], []
    total_loss, n = 0.0, 0

    for batch in loader:
        x = batch["image"].to(device)
        y = batch["label"].long().to(device)

        logits = model(x)

        if criterion is not None:
            loss = criterion(logits, y)
            total_loss += float(loss.item()) * x.size(0)
        n += x.size(0)

        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.detach().cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    preds = probs.argmax(axis=1)

    out = {
        "loss": (total_loss / max(n, 1)) if criterion is not None else float("nan"),
        "acc": float((preds == labels).mean()),
        "probs": probs,
        "labels": labels,
        "preds": preds,
    }

    # Binary extras
    if len(class_names) == 2:
        pos_idx = class_names.index(pos_class_name)
        y_true = (labels == pos_idx).astype(int)
        y_score = probs[:, pos_idx]
        ap = float(skm.average_precision_score(y_true, y_score))

        out.update(
            {
                "pos_idx": int(pos_idx),
                "y_true": y_true,
                "y_score": y_score,
                "ap": ap,
            }
        )

    return out


def metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float = 0.5) -> Dict:
    """
    Compute binary confusion-matrix-derived metrics at a given threshold.

    Returns confusion_matrix in [[TN, FP],[FN, TP]] format plus:
      sensitivity, specificity, ppv, npv, accuracy
    """
    y_hat = (y_score >= thr).astype(int)
    cm = skm.confusion_matrix(y_true, y_hat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else float("nan")

    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "ppv": float(ppv),
        "npv": float(npv),
        "accuracy": float(acc),
        "confusion_matrix": cm,
    }


def pick_threshold_max_f1(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """
    Pick threshold that maximizes F1, computed from PR curve points.

    Returns (threshold, best_f1).
    """
    precision, recall, thr = skm.precision_recall_curve(y_true, y_score)
    # thr length = len(precision) - 1
    f1 = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    best_i = int(np.nanargmax(f1))
    return float(thr[best_i]), float(f1[best_i])


def pick_threshold_target_sens(
    y_true: np.ndarray, y_score: np.ndarray, target_sens: float = 0.95, n_grid: int = 2001
) -> Tuple[float, Dict]:
    """
    Pick the highest threshold that achieves sensitivity >= target_sens.

    Returns (threshold, metrics_dict_at_threshold).
    """
    thr_grid = np.linspace(1.0, 0.0, n_grid)  # high -> low
    for t in thr_grid:
        m = metrics_at_threshold(y_true, y_score, float(t))
        if m["sensitivity"] >= target_sens:
            return float(t), m
    # fallback
    m0 = metrics_at_threshold(y_true, y_score, 0.0)
    return 0.0, m0


def find_errors_binary(
    items,
    y_score: np.ndarray,
    pos_idx: int,
    thr: float,
    topk: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Identify false positives/false negatives indices for a list of items containing 'label' and 'image'.

    items: list of dicts with keys {'image','label'} in original label space (0..K-1).
    y_score: probability for pos_idx (length N).
    pos_idx: integer index of positive class in class_names.
    thr: threshold applied to y_score.
    """
    labels = np.array([it["label"] for it in items], dtype=int)
    y_true = (labels == pos_idx).astype(int)
    y_hat = (y_score >= thr).astype(int)

    fp_idx = np.where((y_true == 0) & (y_hat == 1))[0]
    fn_idx = np.where((y_true == 1) & (y_hat == 0))[0]

    # sort by confidence
    fp_sorted = fp_idx[np.argsort(-y_score[fp_idx])]  # highest scores among negatives
    fn_sorted = fn_idx[np.argsort(y_score[fn_idx])]   # lowest scores among positives

    return {
        "fp_idx": fp_sorted[:topk],
        "fn_idx": fn_sorted[:topk],
    }
