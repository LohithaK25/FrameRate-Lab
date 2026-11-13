from __future__ import annotations

import torch.nn as nn


def build_model(weights: str = "DEFAULT") -> nn.Module:
    """Build the ResNet-18 image workload used for the first benchmark.

    The model architecture is the spatial backbone used in TRIGUARD-X.
    This benchmark intentionally runs one image per inference request; it does
    not claim to reproduce the complete three-stream video classifier.
    """
    try:
        from torchvision.models import ResNet18_Weights, resnet18
    except Exception as exc:  # pragma: no cover - environment-specific import error
        raise RuntimeError(
            "Could not import torchvision ResNet-18. Install compatible torch "
            "and torchvision builds; see the README setup instructions."
        ) from exc

    if weights.upper() == "DEFAULT":
        selected_weights = ResNet18_Weights.DEFAULT
    elif weights.lower() in {"none", "random"}:
        selected_weights = None
    else:
        raise ValueError("weights must be 'DEFAULT' or 'none'")

    model = resnet18(weights=selected_weights)
    model.eval()
    return model
