"""Training losses and batch-level metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .data import IGNORE_INDEX


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    dimensions = tuple(range(1, probability.ndim))
    intersection = (probability * target).sum(dimensions)
    denominator = probability.sum(dimensions) + target.sum(dimensions)
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def masked_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = target != IGNORE_INDEX
    return F.cross_entropy(logits[valid], target[valid]) if valid.any() else logits.sum() * 0.0


def compute_loss(
    outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, float]]:
    components = {
        "seg_bce": F.binary_cross_entropy_with_logits(outputs["segmentation"], batch["mask"]),
        "seg_dice": dice_loss(outputs["segmentation"], batch["mask"]),
        "detection": F.binary_cross_entropy_with_logits(outputs["detection"], batch["detection"]),
        "malignancy": masked_cross_entropy(outputs["malignancy"], batch["malignancy"]),
        "tirads": masked_cross_entropy(outputs["tirads"], batch["tirads"]),
        "extension": masked_cross_entropy(outputs["extension"], batch["extension"]),
        "location": masked_cross_entropy(outputs["location"], batch["location"]),
        "t_clinsig": masked_cross_entropy(outputs["t_clinsig"], batch["t_clinsig"]),
    }
    total = (
        components["seg_bce"] + components["seg_dice"] + 0.20 * components["detection"]
        + 0.10 * components["malignancy"] + 0.10 * components["tirads"]
        + 0.10 * components["extension"] + 0.10 * components["location"]
        + 0.10 * components["t_clinsig"]
    )
    return total, {name: float(value.detach()) for name, value in components.items()}


@torch.no_grad()
def batch_metrics(
    outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> dict[str, tuple[float, int]]:
    prediction = torch.sigmoid(outputs["segmentation"]) >= 0.5
    target = batch["mask"] >= 0.5
    dimensions = tuple(range(1, prediction.ndim))
    intersection = (prediction & target).sum(dimensions).float()
    prediction_sum = prediction.sum(dimensions).float()
    target_sum = target.sum(dimensions).float()
    dice = ((2 * intersection + 1e-6) / (prediction_sum + target_sum + 1e-6)).sum().item()
    union = (prediction | target).sum(dimensions).float()
    iou = ((intersection + 1e-6) / (union + 1e-6)).sum().item()
    count = prediction.shape[0]
    result: dict[str, tuple[float, int]] = {"dice": (dice, count), "iou": (iou, count)}
    detection = (torch.sigmoid(outputs["detection"]) >= 0.5).long()
    result["detection_acc"] = ((detection == batch["detection"].long()).sum().item(), count)
    for name in ("malignancy", "tirads", "extension", "location", "t_clinsig"):
        valid = batch[name] != IGNORE_INDEX
        if valid.any():
            correct = (outputs[name][valid].argmax(1) == batch[name][valid]).sum().item()
            result[f"{name}_acc"] = (correct, int(valid.sum().item()))
    return result
