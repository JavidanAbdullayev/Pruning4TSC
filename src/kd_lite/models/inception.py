"""InceptionTime: Ismail-Fawaz et al., 2019.

The reference implementation uses a uniform filter count per branch. Here
we allow per-block, per-branch filter counts so pruned sub-architectures
can be rebuilt without reshaping hacks.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class InceptionBlock(nn.Module):
    """One Inception module with 3 convolutional branches + maxpool branch."""

    DEFAULT_BOTTLENECK = 32

    def __init__(
        self,
        in_channels: int,
        out_channels: list[int] | tuple[int, int, int, int],
        stride: int = 1,
        use_bottleneck: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = list(out_channels)
        self.use_bottleneck = use_bottleneck and in_channels > 1

        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(
                in_channels, self.DEFAULT_BOTTLENECK, kernel_size=1, padding="same", bias=False
            )
            branch_in = self.DEFAULT_BOTTLENECK
        else:
            self.bottleneck = None
            branch_in = in_channels

        def maybe_conv(out_c: int, k: int) -> nn.Conv1d | None:
            if out_c <= 0:
                return None
            return nn.Conv1d(branch_in, out_c, kernel_size=k, stride=1, padding="same", bias=False)

        self.branch1 = maybe_conv(self.out_channels[0], 40)
        self.branch2 = maybe_conv(self.out_channels[1], 20)
        self.branch3 = maybe_conv(self.out_channels[2], 10)
        self.branch4 = (
            nn.Sequential(
                nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
                nn.Conv1d(in_channels, self.out_channels[3], kernel_size=1, padding="same", bias=False),
            )
            if self.out_channels[3] > 0
            else None
        )

        self.bn = nn.BatchNorm1d(sum(self.out_channels))
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.bottleneck(x) if self.bottleneck is not None else x

        branches = []
        if self.branch1 is not None:
            branches.append(self.branch1(y))
        if self.branch2 is not None:
            branches.append(self.branch2(y))
        if self.branch3 is not None:
            branches.append(self.branch3(y))
        if self.branch4 is not None:
            branches.append(self.branch4(x))

        return self.relu(self.bn(torch.cat(branches, dim=1)))


class InceptionTime(nn.Module):
    """InceptionTime classifier with per-block filter configuration.

    Args:
        input_shape: Shape of one sample (unused, kept for API symmetry).
        n_classes: Number of output classes.
        n_filters: ``(depth, 4)`` array of per-branch filter counts.
            Pass an int to use the default uniform configuration.
        depth: Number of inception modules (default 6).
        residual: Whether to add residual shortcuts every 3 blocks.
    """

    def __init__(
        self,
        input_shape,
        n_classes: int,
        n_filters: np.ndarray | int = 32,
        depth: int = 6,
        residual: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(n_filters, int):
            n_filters = np.full((depth, 4), n_filters, dtype=int)
        else:
            n_filters = np.asarray(n_filters, dtype=int)
            if n_filters.shape != (depth, 4):
                raise ValueError(f"n_filters must have shape ({depth}, 4), got {n_filters.shape}")

        self.depth = depth
        self.residual = residual
        self.n_filters = n_filters
        self.n_classes = n_classes

        self.inception = nn.ModuleList()
        self.shortcut = nn.ModuleList()

        for d in range(depth):
            in_c = 1 if d == 0 else int(n_filters[d - 1].sum())
            self.inception.append(InceptionBlock(in_c, n_filters[d].tolist()))
            if residual and d % 3 == 2:
                shortcut_in = 1 if d == 2 else int(n_filters[2].sum())
                self.shortcut.append(ResidualBlock(shortcut_in, int(n_filters[d].sum())))

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(int(n_filters[-1].sum()), n_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_res = x
        for d in range(self.depth):
            x = self.inception[d](x)
            if self.residual and d % 3 == 2:
                y = self.shortcut[d // 3](input_res)
                x = self.relu(x + y)
                input_res = x
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

    def forward_with_features(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return logits plus per-block feature maps."""
        feats: list[torch.Tensor] = []
        input_res = x
        for d in range(self.depth):
            x = self.inception[d](x)
            if self.residual and d % 3 == 2:
                y = self.shortcut[d // 3](input_res)
                x = self.relu(x + y)
                input_res = x
            feats.append(x)
        out = self.avgpool(x).flatten(1)
        return self.fc(out), feats
