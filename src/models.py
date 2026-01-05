# src/models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn

from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    densenet121,
    DenseNet121_Weights,
)


def freeze_backbone(model: nn.Module, head_prefixes: Tuple[str, ...]) -> None:
    """
    Freeze all parameters except those whose names start with any of head_prefixes.
    Example for ResNet: head_prefixes=("fc.",)
    """
    for name, p in model.named_parameters():
        p.requires_grad = any(name.startswith(pref) for pref in head_prefixes)


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


def get_gradcam_target_layer(model: nn.Module, arch: str) -> nn.Module:
    """
    Return a reasonable last-conv layer for Grad-CAM for the given architecture.
    """
    arch = arch.lower()
    if arch in ("resnet18", "resnet"):
        # last block conv in layer4 is a standard Grad-CAM target
        return model.layer4[-1].conv2
    if arch in ("densenet121", "densenet"):
        # last denseblock's last denselayer conv2 is typically good
        # torchvision DenseNet: features.denseblock4.denselayer16.conv2 exists
        return model.features.denseblock4.denselayer16.conv2

    raise ValueError(f"Unknown arch for gradcam target layer: {arch}")


def build_model(
    arch: str,
    num_classes: int,
    pretrained: bool = True,
    device: str = "cpu",
) -> nn.Module:
    """
    Build a classifier model with replaced head for num_classes.
    Supported: 'resnet18', 'densenet121'
    """
    arch_l = arch.lower()

    if arch_l in ("resnet18", "resnet"):
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif arch_l in ("densenet121", "densenet"):
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    else:
        raise ValueError(f"Unsupported arch: {arch}. Use 'resnet18' or 'densenet121'.")

    return model.to(device)


def get_head_prefixes(arch: str) -> Tuple[str, ...]:
    """
    Return parameter-name prefixes that correspond to the classification head.
    Used for freezing backbone during head-only training.
    """
    arch_l = arch.lower()
    if arch_l in ("resnet18", "resnet"):
        return ("fc.",)
    if arch_l in ("densenet121", "densenet"):
        return ("classifier.",)
    raise ValueError(f"Unsupported arch: {arch}")
