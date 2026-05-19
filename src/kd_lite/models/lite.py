"""LITE: a lightweight time-series classifier with hand-crafted hybrid filters.

Based on Ismail-Fawaz et al., "LITE: Light Inception with boosTing tEchniques
for Time Series Classification". This implementation supports variable
per-layer filter counts so the same module can express both the full
model and any pruned subnetwork.
"""

from __future__ import annotations

import torch
from torch import nn


class HybridBlock(nn.Module):
    """Frozen hand-designed multi-scale filters (positive/negative/peak)."""

    def __init__(
        self,
        input_channels: int = 1,
        kernel_sizes: tuple[int, ...] = (2, 4, 8, 16, 32, 64),
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()

        # Sign-alternating filters (+/- and -/+)
        for sign in (1, -1):
            for k in kernel_sizes:
                w = torch.ones((input_channels, 1, k))
                idx = torch.arange(k)
                mask = (idx % 2 == 0) if sign == 1 else (idx % 2 > 0)
                w[:, :, mask] *= -1
                conv = nn.Conv1d(input_channels, 1, kernel_size=k, padding="same", bias=False)
                with torch.no_grad():
                    conv.weight = nn.Parameter(w, requires_grad=False)
                self.layers.append(conv)

        # Peak-shaped filters
        for k in kernel_sizes[1:]:
            kt = k + k // 2
            w = torch.zeros((kt, input_channels, 1))
            xmash = torch.linspace(0, 1, steps=k // 4 + 1)[1:].reshape((-1, 1, 1))
            left = xmash ** 2
            right = left.flip(0)
            w[0 : k // 4] = -left
            w[k // 4 : k // 2] = -right
            w[k // 2 : 3 * k // 4] = 2 * left
            w[3 * k // 4 : k] = 2 * right
            w[k : 5 * k // 4] = -left
            w[5 * k // 4 :] = -right
            conv = nn.Conv1d(input_channels, 1, kernel_size=kt, padding="same", bias=False)
            with torch.no_grad():
                conv.weight = nn.Parameter(w.permute(2, 1, 0), requires_grad=False)
            self.layers.append(conv)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(torch.cat([conv(x) for conv in self.layers], dim=1))


class InceptionBlock(nn.Module):
    """LITE-style inception block: multi-scale convs + optional hybrid branch."""

    HYBRID_CHANNELS = 17  # = 6 + 6 + 5 fixed filters from HybridBlock

    def __init__(
        self,
        n_filters: list[int],
        kernel_size: int,
        dilation_rate: int = 1,
        stride: int = 1,
        use_hybrid_layer: bool = True,
    ) -> None:
        super().__init__()
        self.use_hybrid_layer = use_hybrid_layer
        n_convs = 3
        kernel_sizes = [kernel_size // (2 ** i) for i in range(n_convs)]

        self.inception_layers = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=1,
                    out_channels=n_filters[i],
                    kernel_size=kernel_sizes[i],
                    stride=stride,
                    padding="same",
                    dilation=dilation_rate,
                    bias=False,
                )
                for i in range(n_convs)
                if n_filters[i] != 0
            ]
        )

        self.hybrid = HybridBlock(input_channels=1) if use_hybrid_layer else None
        total_out = sum(n_filters) + (self.HYBRID_CHANNELS if use_hybrid_layer else 0)
        self.bn = nn.BatchNorm1d(total_out)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = [conv(x) for conv in self.inception_layers]
        out = torch.cat(branches, dim=1)
        if self.hybrid is not None:
            out = torch.cat([out, self.hybrid(x)], dim=1)
        return self.relu(self.bn(out))


class FCNBlock(nn.Module):
    """Depthwise-separable FCN block with batchnorm + ReLU."""

    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        n_filters: int,
        dilation_rate: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.depthwise_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
            dilation=dilation_rate,
            groups=in_channels,
            bias=False,
        )
        self.pointwise_conv = nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(n_filters)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        depth = self.depthwise_conv(x)
        out = self.relu(self.bn(self.pointwise_conv(depth)))
        return out, depth


class LITE(nn.Module):
    """LITE classifier.

    Args:
        length_TS: Length of input time series (unused at construction
            time but kept for API symmetry with the original).
        n_classes: Number of output classes.
        n_filters: ``[[c1a, c1b, c1c], c2, c3]`` — three inception branches
            then two FCN block widths.
        kernel_size: Initial kernel size (default 41).
        use_custom_filters: Toggle the hybrid (hand-crafted) branch.
    """

    def __init__(
        self,
        length_TS: int,
        n_classes: int,
        n_filters: list = ((32, 32, 32), 32, 32),
        kernel_size: int = 41,
        use_custom_filters: bool = True,
    ) -> None:
        super().__init__()
        self.length_TS = length_TS
        self.n_classes = n_classes
        # Allow tuple-of-tuples for cleaner config files
        self.n_filters = [list(n_filters[0]), n_filters[1], n_filters[2]]

        k = kernel_size - 1
        self.inception = InceptionBlock(
            n_filters=self.n_filters[0],
            kernel_size=k,
            use_hybrid_layer=use_custom_filters,
        )
        k //= 2
        hybrid_extra = InceptionBlock.HYBRID_CHANNELS if use_custom_filters else 0
        self.fcn_module1 = FCNBlock(
            in_channels=sum(self.n_filters[0]) + hybrid_extra,
            kernel_size=k,
            n_filters=self.n_filters[1],
            dilation_rate=2,
        )
        self.fcn_module2 = FCNBlock(
            in_channels=self.n_filters[1],
            kernel_size=k // 2,
            n_filters=self.n_filters[2],
            dilation_rate=4,
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.n_filters[2], n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inception(x)
        x, _ = self.fcn_module1(x)
        x, _ = self.fcn_module2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)

    def forward_with_features(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return logits plus the per-stage feature maps (for sparsity loss)."""
        feats: list[torch.Tensor] = []
        x = self.inception(x)
        feats.append(x)
        x, _ = self.fcn_module1(x)
        feats.append(x)
        x, _ = self.fcn_module2(x)
        feats.append(x)

        out = self.avgpool(x)
        out = torch.flatten(out, start_dim=1)
        return self.fc(out), feats
