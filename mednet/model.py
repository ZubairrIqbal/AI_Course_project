"""MedNet architecture, copied one-to-one from the reproduction notebook.

Module and attribute names are deliberately unchanged so that checkpoints saved
by the notebook (`torch.save(model.state_dict())`) load without any key mapping.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, ratio: int = 16):
        super().__init__()
        hidden = max(1, channels // ratio)
        self.mlp = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(), nn.Linear(hidden, channels))

    def forward(self, x):
        avg = x.mean((2, 3))
        maximum = x.amax((2, 3))
        weights = torch.sigmoid(self.mlp(avg) + self.mlp(maximum))
        return x * weights[:, :, None, None]


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        maximum = x.amax(dim=1, keepdim=True)
        weights = torch.sigmoid(self.conv(torch.cat([avg, maximum], dim=1)))
        return x * weights


class CBAM(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channel = ChannelAttention(channels)
        self.spatial = SpatialAttention()

    def forward(self, x):
        return self.spatial(self.channel(x))


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(input_channels, input_channels, 3, stride=stride,
                                   padding=1, groups=input_channels, bias=False)
        self.pointwise = nn.Conv2d(input_channels, output_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))


class ResidualDSCBAMBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1, use_cbam: bool = True):
        super().__init__()
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(input_channels, output_channels, stride),
            DepthwiseSeparableConv(output_channels, output_channels),
        )
        self.cbam = CBAM(output_channels) if use_cbam else nn.Identity()
        self.shortcut = nn.Identity()
        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(output_channels),
            )

    def forward(self, x):
        return self.cbam(self.conv(x)) + self.shortcut(x)


class MedNet(nn.Module):
    """Five residual depthwise-separable stages (+CBAM) -> GAP -> FC(256) -> FC(classes)."""

    STAGE_CHANNELS = (64, 128, 256, 512, 1024)

    def __init__(self, num_classes: int, use_cbam: bool = True):
        super().__init__()
        self.stage1 = ResidualDSCBAMBlock(3, 64, 1, use_cbam)
        self.stage2 = ResidualDSCBAMBlock(64, 128, 2, use_cbam)
        self.stage3 = ResidualDSCBAMBlock(128, 256, 2, use_cbam)
        self.stage4 = ResidualDSCBAMBlock(256, 512, 2, use_cbam)
        self.stage5 = ResidualDSCBAMBlock(512, 1024, 2, use_cbam)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, num_classes)

    @property
    def stages(self) -> list[nn.Module]:
        return [self.stage1, self.stage2, self.stage3, self.stage4, self.stage5]

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(self.fc1(x))
        return self.fc2(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
