#!/usr/bin/env python
# coding: utf-8

# ### 1. Imports + path setup

# In[1]:


import sys
from pathlib import Path

ROOT = Path.cwd().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

print("Project root:", ROOT)


# ### 2. Load config and rebuild data/loaders

# In[2]:


from pathlib import Path
from src.utils import load_json
from src.config import Config, seed_everything

RUNS_ROOT = Path("outputs") / "runs"

# 1) Read latest pointer (created by 01_train)
latest_meta = load_json(str(RUNS_ROOT / "_latest_run.json"))

# Common patterns: latest_meta["run_dir"] might be absolute or relative.
run_dir = Path(latest_meta["run_dir"])
if not run_dir.is_absolute():
    run_dir = (Path.cwd() / run_dir).resolve()

# 2) Load the pure config from that run folder
cfg_dict = load_json(str(run_dir / "config.json"))
cfg = Config(**cfg_dict)

# 3) Optionally attach run_dir for convenience
cfg.run_dir = str(run_dir)

seed_everything(cfg.seed, cfg.deterministic)
print("Using run_dir:", run_dir)
print(cfg)


# In[3]:


from src.data import build_datasets, build_loaders

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
    use_weighted_sampler=cfg.use_weighted_sampler,
)

train_loader = loaders["train_loader"]
val_loader   = loaders["val_loader"]
test_loader  = loaders["test_loader"]

train_items = ds["train_items"]
val_items   = ds["val_items"]
test_items  = ds["test_items"]


# ### 3. Load model + best checkpoint

# In[5]:


import torch
import torch.nn as nn

from src.models import build_model, get_gradcam_target_layer

arch = "resnet18"  # must match what you trained with; later you can store in config if desired
model = build_model(arch=arch, num_classes=len(cfg.class_names), pretrained=False, device=cfg.device)

best_ckpt_path = str(Path(run_dir) / "best.pt")
state = torch.load(best_ckpt_path, map_location="cpu")
model.load_state_dict(state)
model.to(cfg.device)
model.eval()

criterion = nn.CrossEntropyLoss()
target_layer = get_gradcam_target_layer(model, arch)

print("Loaded checkpoint:", best_ckpt_path)
print("Grad-CAM target layer:", target_layer)


# ### 4. Evaluate VAL/TEST + thresholding + hard errors

# In[6]:


import numpy as np
import sklearn.metrics as skm

from src.eval import (
    eval_split,
    metrics_at_threshold,
    pick_threshold_max_f1,
    pick_threshold_target_sens,
    find_errors_binary,
)

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

# Thresholds from VAL
thr_f1, best_f1 = pick_threshold_max_f1(val_out["y_true"], val_out["y_score"])
m_val_f1 = metrics_at_threshold(val_out["y_true"], val_out["y_score"], thr_f1)

TARGET_SENS = 0.95
thr_sens, m_val_sens = pick_threshold_target_sens(val_out["y_true"], val_out["y_score"], target_sens=TARGET_SENS)

print("\n=== Thresholds selected on VAL ===")
print(f"Max-F1 threshold: {thr_f1:.4f} (F1={best_f1:.3f})")
print("VAL metrics @ thr_f1:", {k: v for k, v in m_val_f1.items() if k != "confusion_matrix"})

print(f"\nTarget-sensitivity threshold: {thr_sens:.4f} (target sens >= {TARGET_SENS})")
print("VAL metrics @ thr_sens:", {k: v for k, v in m_val_sens.items() if k != "confusion_matrix"})

# Apply thresholds to TEST
m_test_f1 = metrics_at_threshold(test_out["y_true"], test_out["y_score"], thr_f1)
m_test_sens = metrics_at_threshold(test_out["y_true"], test_out["y_score"], thr_sens)

print("\n=== TEST metrics using thresholds chosen on VAL ===")
print(f"\n[TEST @ Max-F1 thr={thr_f1:.4f}]")
print({k: v for k, v in m_test_f1.items() if k != "confusion_matrix"})

print(f"\n[TEST @ TargetSens thr={thr_sens:.4f} (target sens {TARGET_SENS})]")
print({k: v for k, v in m_test_sens.items() if k != "confusion_matrix"})

print("\nConfusion matrix format: [[TN, FP],[FN, TP]]")
print("\nTEST confusion @ thr_f1:\n", m_test_f1["confusion_matrix"])
print("\nTEST confusion @ thr_sens:\n", m_test_sens["confusion_matrix"])

# Hard errors (use Max-F1 threshold by default)
errs = find_errors_binary(
    items=test_items,
    y_score=test_out["y_score"],
    pos_idx=test_out["pos_idx"],
    thr=thr_f1,
    topk=10,
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


# ### 5. Grad-CAM: display a few and save batches to disk

# In[9]:


from src.interpret import GradCAM, show_gradcam, save_gradcam_batch, infer_true_label_from_path

gradcam = GradCAM(model, target_layer)

# Show a couple interactively
print("=== Show Grad-CAM: first 2 FPs ===")
for i in fp_idx[:2]:
    p = test_items[i]["image"]
    show_gradcam(
        model=model,
        gradcam=gradcam,
        path=p,
        class_names=cfg.class_names,
        pos_class_name=cfg.pos_class_name,
        device=cfg.device,
        image_size=cfg.image_size,
        true_label_name=infer_true_label_from_path(p),
    )

print("\n=== Show Grad-CAM: first 1 FN ===")
for i in fn_idx[:1]:
    p = test_items[i]["image"]
    show_gradcam(
        model=model,
        gradcam=gradcam,
        path=p,
        class_names=cfg.class_names,
        pos_class_name=cfg.pos_class_name,
        device=cfg.device,
        image_size=cfg.image_size,
        true_label_name=infer_true_label_from_path(p),
    )

# Save panels to disk under the run directory
out_dir = Path(run_dir) / "gradcam"
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


# ### 6. (Optional) Save evaluation summary JSON

# In[11]:


from src.utils import save_json

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


# In[12]:


fp_paths = [test_items[i]["image"] for i in fp_idx[:10]]
fn_paths = [test_items[i]["image"] for i in fn_idx[:10]]

out_dir = r"medimg_baseline_cls\outputs\gradcam_test_errors"
records_fp = save_gradcam_batch(model, gradcam, fp_paths, out_dir + r"\FP", cfg.class_names, cfg.device, cfg.image_size)
records_fn = save_gradcam_batch(model, gradcam, fn_paths, out_dir + r"\FN", cfg.class_names, cfg.device, cfg.image_size)

print("Saved FP Grad-CAM panels:", len(records_fp))
print("Saved FN Grad-CAM panels:", len(records_fn))


# In[ ]:




