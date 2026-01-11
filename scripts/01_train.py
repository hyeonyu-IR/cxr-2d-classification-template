#!/usr/bin/env python
# coding: utf-8

# ### 1. Imports + Path Setup

# In[1]:


import sys
from pathlib import Path

# notebooks/ -> project root
ROOT = Path.cwd().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

print("Project root:", ROOT)


# ### 2. Config + reproducibility

# In[3]:


data_dir = r"C:\Users\hyeon\Documents\miniconda_medimg_env\data\chest_xray"


# In[4]:


from src.config import Config, seed_everything, ensure_dirs
from src.utils import env_report, save_json

cfg = Config(
    project_name="medimg_baseline_cls",
    data_root=data_dir,
    output_root="outputs",
    image_size=(224, 224),
    batch_size=32,
    num_workers=0,        # start 0 on Windows; increase to 2 or 4 once stable
    pin_memory=True,
    max_epochs=10,
    head_epochs=2,
    lr_head=3e-3,
    lr_finetune=1e-3,     # consider 3e-4 later for smoother FT
    weight_decay=1e-4,
    rebuild_balanced_val=True,
    val_n_per_class=200,
    use_weighted_sampler=True,
    seed=42,
)

seed_everything(cfg.seed, cfg.deterministic)
ensure_dirs(cfg)

print(cfg)
print("Env:", env_report())


# ### 3. Data (items, datasets, loaders)

# In[5]:


from src.data import build_datasets, build_loaders, label_counts

ds = build_datasets(
    root_dir=cfg.data_root,
    class_names=cfg.class_names,
    image_size=cfg.image_size,
    rebuild_balanced_val=cfg.rebuild_balanced_val,
    val_n_per_class=cfg.val_n_per_class,
    seed=cfg.seed,
)

print("Counts:")
print("train:", label_counts(ds["train_items"], len(cfg.class_names)))
print("val  :", label_counts(ds["val_items"], len(cfg.class_names)))
print("test :", label_counts(ds["test_items"], len(cfg.class_names)))

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

# Sanity batch
b = next(iter(loaders["train_loader"]))
print("Batch image shape:", b["image"].shape)
print("Batch labels:", {int(k): int((b["label"].numpy()==k).sum()) for k in set(b["label"].numpy().tolist())})


# ### 4. Model + freeze backbone

# In[6]:


from src.models import build_model, freeze_backbone, get_head_prefixes, get_gradcam_target_layer

arch = "resnet18"  # swap to 'densenet121' later if desired
model = build_model(arch=arch, num_classes=len(cfg.class_names), pretrained=True, device=cfg.device)

# Freeze backbone for head-only stage
freeze_backbone(model, head_prefixes=get_head_prefixes(arch))
print("Trainable tensors:", sum(p.requires_grad for p in model.parameters()), "/", len(list(model.parameters())))

# Save gradcam target layer reference for later
target_layer = get_gradcam_target_layer(model, arch)
print("Grad-CAM target layer:", target_layer)


# ### 5. Train (head-only → fine-tune) + save run metadata

# In[7]:


from src.train import run_training
from src.utils import save_json

result = run_training(
    cfg=cfg,
    model=model,
    train_loader=loaders["train_loader"],
    val_loader=loaders["val_loader"],
    test_loader=loaders["test_loader"],
)

print("Run saved to:", result["run_dir"])
print("Best checkpoint:", result["best_ckpt_path"])
print("VAL summary:", result["val_summary"])
print("TEST summary:", result["test_summary"])

# Save environment report into the run directory for reproducibility
save_json(env_report(), str(Path(result["run_dir"]) / "env_report.json"))

# (Optional) Keep these in memory for the next notebook
RUN_DIR = result["run_dir"]
BEST_CKPT = result["best_ckpt_path"]
print("RUN_DIR =", RUN_DIR)
print("BEST_CKPT =", BEST_CKPT)


# In[8]:


from pathlib import Path
from src.utils import save_json

# Save a pointer to the latest run for downstream notebooks
latest_path = Path("outputs") / "runs" / "_latest_run.json"
save_json(
    {"run_dir": result["run_dir"], "best_ckpt_path": result["best_ckpt_path"]},
    str(latest_path)
)

print("Wrote latest run pointer to:", latest_path)


# In[ ]:




