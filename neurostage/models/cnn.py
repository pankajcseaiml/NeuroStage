"""
Lightweight 1D-CNN Feature Extractor using Depthwise-Separable Convolutions.

Architecture:
    Multi-resolution CNN block:
      For each kernel size k in [50, 25, 10] (at 100 Hz: 500ms, 250ms, 100ms windows):
        → DepthwiseSeparableConv1d → BatchNorm → GELU → MaxPool
      → Concatenate all branch outputs
      → Pointwise conv to fuse → embedding_dim output

Rationale for Depthwise-Separable:
    A standard Conv1d with C_in channels, C_out filters, kernel K has
    C_in × C_out × K parameters.  Depthwise-separable splits it into:
      depthwise: C_in × 1 × K   (one filter per input channel)
      pointwise: C_in × C_out × 1
    Total: C_in×K + C_in×C_out — typically 8–9× fewer MACs for typical sizes.
    This is the primary source of NeuroStage's parameter reduction vs. standard CNNs.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise separable 1D convolution.
    Depthwise: groups=in_channels (one filter per channel).
    Pointwise: 1×1 conv to mix channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        bias: bool = False,
    ):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class CNNBranch(nn.Module):
    """
    One multi-resolution CNN branch for a given kernel size.
    Input : (B, 1, T)  — single EEG channel, T samples
    Output: (B, base_channels, T')
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        kernel_size: int,
        use_depthwise_separable: bool = True,
        dropout: float = 0.25,
    ):
        super().__init__()
        padding = kernel_size // 2  # same-padding

        if use_depthwise_separable:
            self.conv = nn.Sequential(
                DepthwiseSeparableConv1d(in_channels, base_channels, kernel_size, padding=padding),
                nn.BatchNorm1d(base_channels),
                nn.GELU(),
                nn.Dropout(dropout),
                DepthwiseSeparableConv1d(base_channels, base_channels, kernel_size, padding=padding),
                nn.BatchNorm1d(base_channels),
                nn.GELU(),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, base_channels, kernel_size, padding=padding, bias=False),
                nn.BatchNorm1d(base_channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(base_channels, base_channels, kernel_size, padding=padding, bias=False),
                nn.BatchNorm1d(base_channels),
                nn.GELU(),
            )

        self.pool = nn.AdaptiveAvgPool1d(32)   # Downsample each branch to fixed length 32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)   # (B, base_channels, T)
        x = self.pool(x)   # (B, base_channels, 32)
        return x


class EEGFeatureExtractor(nn.Module):
    """
    Multi-resolution 1D-CNN feature extractor.

    Input  : (B, 1, T)  where T = sfreq × epoch_sec (e.g. 100 × 30 = 3000 samples)
    Output : (B, embedding_dim)

    Three parallel branches (kernel sizes 50/25/10 at 100 Hz ≈ 500ms/250ms/100ms)
    capture slow delta/spindle patterns and fast micro-structure respectively.
    Outputs are concatenated and projected to embedding_dim via a pointwise conv.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        kernel_sizes: list[int] | None = None,
        use_depthwise_separable: bool = True,
        dropout: float = 0.25,
        embedding_dim: int = 128,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [50, 25, 10]

        self.branches = nn.ModuleList([
            CNNBranch(in_channels, base_channels, k, use_depthwise_separable, dropout)
            for k in kernel_sizes
        ])

        # Fuse all branch outputs
        fused_channels = base_channels * len(kernel_sizes) * 32  # flat after pool
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fused_channels, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 1, T)
        returns: (B, embedding_dim)
        """
        branch_outs = [branch(x) for branch in self.branches]   # each: (B, C, 32)
        fused = torch.cat(branch_outs, dim=1)                    # (B, C*n_branches, 32)
        return self.projection(fused)                             # (B, embedding_dim)

    @property
    def output_dim(self) -> int:
        return self.projection[-2].out_features
