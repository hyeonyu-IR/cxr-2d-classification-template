# src/interpret.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image


def preprocess_for_model(
    path: str,
    image_size: Tuple[int, int] = (224, 224),
    device: str = "cpu",
) -> Tuple[Image.Image, torch.Tensor]:
    """
    Load an image as RGB, resize, convert to float tensor in [0,1], shape [1,3,H,W].
    This matches the Phase-1 pipeline you trained with (no ImageNet mean/std normalization).
    """
    img = Image.open(path).convert("RGB")
    img = img.resize(image_size)

    x = np.array(img).astype(np.float32) / 255.0  # [H,W,3] in [0,1]
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    return img, x.to(device)


class GradCAM:
    """
    Grad-CAM for CNN classifiers.

    Usage:
      gradcam = GradCAM(model, target_layer)
      out = gradcam(x, class_idx=None)
      cam = out["cam"]  # [H',W'] normalized to [0,1]
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove(self):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles = []

    def __call__(self, x: torch.Tensor, class_idx: Optional[int] = None) -> Dict:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        logits = self.model(x)  # [1, K]
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        score = logits[:, class_idx].sum()
        score.backward()

        A = self.activations  # [1, C, H', W']
        dA = self.gradients   # [1, C, H', W']

        if A is None or dA is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients. Check target_layer.")

        weights = dA.mean(dim=(2, 3), keepdim=True)     # [1, C, 1, 1]
        cam = (weights * A).sum(dim=1, keepdim=True)    # [1, 1, H', W']
        cam = F.relu(cam)

        cam = cam.squeeze().detach().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-12)

        probs = torch.softmax(logits, dim=1).detach().cpu().numpy().squeeze()
        pred_idx = int(np.argmax(probs))

        return {
            "cam": cam,
            "probs": probs,
            "pred_idx": pred_idx,
            "class_idx_used": int(class_idx),
            "logits": logits.detach().cpu().numpy().squeeze(),
        }


def overlay_cam_on_image(
    pil_img: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.35,
    cmap_name: str = "jet",
) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on an RGB image.

    Returns overlay as float RGB array in [0,1] of shape [H,W,3].
    """
    w, h = pil_img.size
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize((w, h), resample=Image.BILINEAR)
    cam_np = np.array(cam_img).astype(np.float32) / 255.0

    img_np = np.array(pil_img).astype(np.float32) / 255.0
    cmap = plt.get_cmap(cmap_name)
    heat = cmap(cam_np)[:, :, :3]

    overlay = (1 - alpha) * img_np + alpha * heat
    overlay = np.clip(overlay, 0, 1)
    return overlay


def show_gradcam(
    model: torch.nn.Module,
    gradcam: GradCAM,
    path: str,
    class_names: Tuple[str, ...],
    pos_class_name: str,
    device: str,
    image_size: Tuple[int, int] = (224, 224),
    true_label_name: Optional[str] = None,
    class_idx: Optional[int] = None,
    alpha: float = 0.35,
):
    """
    Display original, heatmap, and overlay. Also prints class probabilities.
    """
    pil_img, x = preprocess_for_model(path, image_size=image_size, device=device)
    out = gradcam(x, class_idx=class_idx)

    cam = out["cam"]
    overlay = overlay_cam_on_image(pil_img, cam, alpha=alpha)

    probs = out["probs"]
    pred_idx = out["pred_idx"]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(pil_img)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(cam, cmap="jet")
    plt.title("Grad-CAM heatmap")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    title = f"Overlay | pred={class_names[pred_idx]} (p={probs[pred_idx]:.3f})"
    if true_label_name is not None:
        title += f" | true={true_label_name}"
    plt.title(title)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print("Path:", path)
    print("Probabilities:", {class_names[i]: float(probs[i]) for i in range(len(class_names))})


def save_gradcam_figure(
    model: torch.nn.Module,
    gradcam: GradCAM,
    path: str,
    out_png: str,
    class_names: Tuple[str, ...],
    device: str,
    image_size: Tuple[int, int] = (224, 224),
    true_label_name: Optional[str] = None,
    alpha: float = 0.35,
    dpi: int = 160,
):
    """
    Save a single Grad-CAM panel (original, heatmap, overlay) to a PNG.
    """
    out_png = str(out_png)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)

    pil_img, x = preprocess_for_model(path, image_size=image_size, device=device)
    out = gradcam(x, class_idx=None)

    cam = out["cam"]
    overlay = overlay_cam_on_image(pil_img, cam, alpha=alpha)

    probs = out["probs"]
    pred_idx = out["pred_idx"]

    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(pil_img)
    ax1.set_title("Original")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(cam, cmap="jet")
    ax2.set_title("Grad-CAM heatmap")
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(overlay)
    title = f"Overlay | pred={class_names[pred_idx]} (p={probs[pred_idx]:.3f})"
    if true_label_name is not None:
        title += f" | true={true_label_name}"
    ax3.set_title(title)
    ax3.axis("off")

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return {
        "path": path,
        "out_png": out_png,
        "pred": class_names[pred_idx],
        "pred_prob": float(probs[pred_idx]),
        "probs": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
    }


def infer_true_label_from_path(path: str) -> Optional[str]:
    """
    Kaggle-style convenience: infer true label name from folder.
    Returns 'PNEUMONIA', 'NORMAL', or None.
    """
    p = path.replace("/", "\\").upper()
    if "\\PNEUMONIA\\" in p:
        return "PNEUMONIA"
    if "\\NORMAL\\" in p:
        return "NORMAL"
    return None


def save_gradcam_batch(
    model: torch.nn.Module,
    gradcam: GradCAM,
    paths: List[str],
    out_dir: str,
    class_names: Tuple[str, ...],
    device: str,
    image_size: Tuple[int, int] = (224, 224),
    alpha: float = 0.35,
) -> List[Dict]:
    """
    Save Grad-CAM panels for multiple images to out_dir. Returns list of metadata dicts.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for i, p in enumerate(paths):
        true_label = infer_true_label_from_path(p)
        out_png = out_dir / f"{i:03d}_{Path(p).stem}.png"
        rec = save_gradcam_figure(
            model=model,
            gradcam=gradcam,
            path=p,
            out_png=str(out_png),
            class_names=class_names,
            device=device,
            image_size=image_size,
            true_label_name=true_label,
            alpha=alpha,
        )
        records.append(rec)
    return records
