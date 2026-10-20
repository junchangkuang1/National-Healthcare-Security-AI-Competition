"""ThyroidXL network definition."""

from __future__ import annotations

from typing import Optional

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class ThyroidXL(nn.Module):
    """U-Net++/EfficientNet-B5 segmentation with six auxiliary task heads."""

    def __init__(self, encoder_weights: Optional[str] = "imagenet"):
        super().__init__()
        self.backbone = smp.UnetPlusPlus(
            encoder_name="timm-efficientnet-b5",
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            activation=None,
        )
        channels = self.backbone.encoder.out_channels[-1]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.25)
        self.heads = nn.ModuleDict(
            {
                "detection": nn.Linear(channels, 1),
                "malignancy": nn.Linear(channels, 2),
                "tirads": nn.Linear(channels, 5),
                "extension": nn.Linear(channels, 4),
                "location": nn.Linear(channels, 7),
                "t_clinsig": nn.Linear(channels, 2),
            }
        )

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone.encoder(inputs)
        decoder = self.backbone.decoder(features)
        outputs = {"segmentation": self.backbone.segmentation_head(decoder)}
        pooled = self.dropout(self.pool(features[-1]).flatten(1))
        for name, head in self.heads.items():
            outputs[name] = head(pooled)
        outputs["detection"] = outputs["detection"].squeeze(1)
        return outputs
