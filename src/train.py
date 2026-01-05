# src/train.py
from __future__ import annotations

import time
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .eval import eval_split


def _save_json(obj: Dict[str, Any], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)


def train_one_epoch(
    model: nn.Module,
    loader,
    device: str,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    desc: str = "Train",
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    n = 0

    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        x = batch["image"].to(device)
        y = batch["label"].long().to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * x.size(0)
        n += x.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return {"loss": total_loss / max(n, 1)}


@torch.no_grad()
def quick_train_stats(model: nn.Module, loader, device: str, max_batches: int = 10) -> Dict[str, Any]:
    """
    Quick diagnostics to catch collapse: sample a few batches and compute
    approximate train accuracy + predicted class counts.
    """
    model.eval()
    preds_all, labels_all = [], []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x = batch["image"].to(device)
        y = batch["label"].long().to(device)
        logits = model(x)
        preds = logits.argmax(1)
        preds_all.append(preds.detach().cpu().numpy())
        labels_all.append(y.detach().cpu().numpy())

    preds_all = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)
    acc = float((preds_all == labels_all).mean())

    pred_counts = {int(k): int((preds_all == k).sum()) for k in np.unique(preds_all)}
    return {"train_acc_quick": acc, "train_pred_counts": pred_counts}


def run_training(
    cfg,
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    run_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Two-stage training:
      Stage A: head-only for cfg.head_epochs at cfg.lr_head
      Stage B: full fine-tune for remaining epochs at cfg.lr_finetune

    Saves best checkpoint by:
      (val_ap, -val_loss, val_acc) lexicographic

    Returns dict with paths and history.
    """
    # Resolve run directory
    if run_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_dir = str(Path(cfg.output_root) / "runs" / f"{cfg.project_name}_{ts}")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    _save_json(asdict(cfg), str(run_dir / "config.json"))

    criterion = nn.CrossEntropyLoss()

    history: List[Dict[str, Any]] = []
    best_score = None
    best_state = None

    def consider_best(row: Dict[str, Any]):
        nonlocal best_score, best_state
        val_ap = row.get("val_ap", float("nan"))
        val_loss = row.get("val_loss", float("inf"))
        val_acc = row.get("val_acc", float("nan"))
        score = (val_ap, -val_loss, val_acc)

        if best_score is None or score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # -----------------------
    # Stage A: head-only
    # -----------------------
    print("\n=== Stage A: Head-only training ===")
    if hasattr(cfg, "lr_head"):
        lr_head = cfg.lr_head
    else:
        lr_head = 3e-3

    # Optimizer should include only trainable params (head frozen upstream)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_head,
        weight_decay=getattr(cfg, "weight_decay", 1e-4),
    )

    for epoch in range(1, cfg.head_epochs + 1):
        print(f"\n[Epoch {epoch}/{cfg.max_epochs}] (Head-only)")
        t0 = time.time()

        tr = train_one_epoch(model, train_loader, cfg.device, optimizer, criterion, desc="Train (head)")
        val_out = eval_split(
            model=model,
            loader=val_loader,
            device=cfg.device,
            criterion=criterion,
            class_names=cfg.class_names,
            pos_class_name=cfg.pos_class_name,
        )
        test_out = eval_split(
            model=model,
            loader=test_loader,
            device=cfg.device,
            criterion=criterion,
            class_names=cfg.class_names,
            pos_class_name=cfg.pos_class_name,
        )

        q = quick_train_stats(model, train_loader, cfg.device, max_batches=10)
        dt = time.time() - t0

        row = {
            "epoch": epoch,
            "stage": "head",
            "train_loss": tr["loss"],
            **q,
            "val_loss": val_out["loss"],
            "val_acc": val_out["acc"],
            "val_ap": val_out.get("ap", float("nan")),
            "test_ap": test_out.get("ap", float("nan")),
            "sec": dt,
        }
        history.append(row)
        consider_best(row)

        print(
            f"[Head] tr_loss={row['train_loss']:.4f} tr_acc~={row['train_acc_quick']:.3f} preds={row['train_pred_counts']} | "
            f"val_loss={row['val_loss']:.4f} val_acc={row['val_acc']:.3f} val_AP={row['val_ap']:.3f} | "
            f"test_AP={row['test_ap']:.3f} | {dt:.1f}s"
        )

    # -----------------------
    # Stage B: full fine-tune
    # -----------------------
    print("\n=== Stage B: Full fine-tuning ===")
    # Unfreeze upstream or here; for safety, ensure all params trainable
    for p in model.parameters():
        p.requires_grad = True

    lr_ft = getattr(cfg, "lr_finetune", 1e-3)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr_ft,
        weight_decay=getattr(cfg, "weight_decay", 1e-4),
    )

    for epoch in range(cfg.head_epochs + 1, cfg.max_epochs + 1):
        print(f"\n[Epoch {epoch}/{cfg.max_epochs}] (Fine-tuning)")
        t0 = time.time()

        tr = train_one_epoch(model, train_loader, cfg.device, optimizer, criterion, desc="Train (ft)")
        val_out = eval_split(
            model=model,
            loader=val_loader,
            device=cfg.device,
            criterion=criterion,
            class_names=cfg.class_names,
            pos_class_name=cfg.pos_class_name,
        )
        test_out = eval_split(
            model=model,
            loader=test_loader,
            device=cfg.device,
            criterion=criterion,
            class_names=cfg.class_names,
            pos_class_name=cfg.pos_class_name,
        )

        q = quick_train_stats(model, train_loader, cfg.device, max_batches=10)
        dt = time.time() - t0

        row = {
            "epoch": epoch,
            "stage": "finetune",
            "train_loss": tr["loss"],
            **q,
            "val_loss": val_out["loss"],
            "val_acc": val_out["acc"],
            "val_ap": val_out.get("ap", float("nan")),
            "test_ap": test_out.get("ap", float("nan")),
            "sec": dt,
        }
        history.append(row)
        consider_best(row)

        print(
            f"[FT ] tr_loss={row['train_loss']:.4f} tr_acc~={row['train_acc_quick']:.3f} preds={row['train_pred_counts']} | "
            f"val_loss={row['val_loss']:.4f} val_acc={row['val_acc']:.3f} val_AP={row['val_ap']:.3f} | "
            f"test_AP={row['test_ap']:.3f} | {dt:.1f}s"
        )

    # Save history
    df = pd.DataFrame(history)
    df.to_csv(run_dir / "history.csv", index=False)

    # Save best checkpoint
    if best_state is not None:
        torch.save(best_state, run_dir / "best.pt")

    # Reload best for final summaries
    if best_state is not None:
        model.load_state_dict(best_state)

    final_val = eval_split(
        model=model,
        loader=val_loader,
        device=cfg.device,
        criterion=criterion,
        class_names=cfg.class_names,
        pos_class_name=cfg.pos_class_name,
    )
    final_test = eval_split(
        model=model,
        loader=test_loader,
        device=cfg.device,
        criterion=criterion,
        class_names=cfg.class_names,
        pos_class_name=cfg.pos_class_name,
    )

    # Save summaries (lightweight)
    val_summary = {
        "loss": float(final_val["loss"]),
        "acc": float(final_val["acc"]),
        "ap": float(final_val.get("ap", float("nan"))),
    }
    test_summary = {
        "loss": float(final_test["loss"]),
        "acc": float(final_test["acc"]),
        "ap": float(final_test.get("ap", float("nan"))),
    }
    _save_json(val_summary, str(run_dir / "val_summary.json"))
    _save_json(test_summary, str(run_dir / "test_summary.json"))

    return {
        "run_dir": str(run_dir),
        "best_score": best_score,
        "history": history,
        "val_summary": val_summary,
        "test_summary": test_summary,
        "best_ckpt_path": str(run_dir / "best.pt"),
    }
