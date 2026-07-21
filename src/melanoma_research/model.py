"""Checkpoint-compatible ResNeXt50-CBAM model definition.

Install the optional ``model`` dependencies before importing this module.
"""

from __future__ import annotations

from types import MethodType

import timm
import torch
from torch import nn
from torch.nn import functional as F


class CBAM(nn.Module):
    """Convolutional Block Attention Module with channel and spatial attention."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        if channels < reduction:
            raise ValueError("channels must be at least as large as reduction")
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        attended = inputs * self.channel_att(inputs)
        spatial = torch.cat(
            (
                torch.mean(attended, dim=1, keepdim=True),
                torch.max(attended, dim=1, keepdim=True).values,
            ),
            dim=1,
        )
        return attended * self.spatial_att(spatial)


def _forward_with_cbam(self: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Match the forward path used to produce the archived research checkpoint."""

    identity = inputs
    output = F.relu(self.bn1(self.conv1(inputs)))
    output = F.relu(self.bn2(self.conv2(output)))
    output = self.bn3(self.conv3(output))
    output = self.cbam(output)
    if self.downsample is not None:
        identity = self.downsample(inputs)
    return F.relu(output + identity)


def inject_cbam(model: nn.Module, stages: tuple[str, ...] = ("layer3", "layer4")) -> nn.Module:
    """Insert attention before the residual addition in selected ResNeXt stages."""

    for stage_name in stages:
        stage = getattr(model, stage_name, None)
        if stage is None:
            raise ValueError(f"model does not expose stage {stage_name!r}")
        for block in stage:
            if not hasattr(block, "bn3"):
                raise TypeError(f"stage {stage_name!r} contains a non-bottleneck block")
            block.cbam = CBAM(block.bn3.num_features)
            block.forward = MethodType(_forward_with_cbam, block)
    return model


class BinaryResNeXtCBAM(nn.Module):
    """Binary ResNeXt50 classifier with CBAM in stages three and four."""

    def __init__(self, *, pretrained: bool = False) -> None:
        super().__init__()
        self.model = inject_cbam(
            timm.create_model("resnext50_32x4d", pretrained=pretrained)
        )
        features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.model.forward_features(inputs)
        pooled = self.model.global_pool(features).flatten(1)
        return self.model.fc(pooled)


def build_model(*, pretrained: bool = False) -> BinaryResNeXtCBAM:
    """Build the architecture used by the archived binary checkpoint."""

    return BinaryResNeXtCBAM(pretrained=pretrained)
