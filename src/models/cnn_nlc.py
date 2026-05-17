# SPDX-License-Identifier: MIT
"""
CNN-based Nonlinear Compensation model.

Uses 1D convolutions to capture the temporal structure of fiber-induced
distortion. Chromatic dispersion spreads energy across neighboring symbols,
making CNN a natural fit for learning the inverse mapping.

Architecture:
- 1D Conv layers with increasing channels
- Residual connections
- Global average pooling → dense output
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CNN_NLC(nn.Module):
    """
    1D CNN for nonlinear compensation.

    Input: complex signal → split into I/Q channels (2 channels, 1D)
    Output: compensated I/Q signal (2 channels)

    Args:
        in_channels: input channels (2 for I/Q)
        out_channels: output channels (2 for I/Q)
        hidden_channels: list of hidden channel sizes for conv layers
        kernel_size: convolution kernel size
        num_blocks: number of residual blocks
        dropout: dropout probability
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        hidden_channels: list[int] = [64, 128, 128, 64],
        kernel_size: int = 7,
        num_blocks: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # Initial convolution
        self.conv_in = nn.Conv1d(
            in_channels, hidden_channels[0],
            kernel_size=kernel_size, padding=self.padding
        )
        self.bn_in = nn.BatchNorm1d(hidden_channels[0])
        self.act_in = nn.GELU()

        # Residual blocks
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            cin = hidden_channels[i % len(hidden_channels)]
            cout = hidden_channels[(i + 1) % len(hidden_channels)]
            self.blocks.append(nn.Sequential(
                nn.Conv1d(cin, cout, kernel_size=kernel_size, padding=self.padding),
                nn.BatchNorm1d(cout),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(cout, cout, kernel_size=kernel_size, padding=self.padding),
                nn.BatchNorm1d(cout),
            ))

        # Final convolution
        self.conv_out = nn.Conv1d(
            hidden_channels[(num_blocks) % len(hidden_channels)],
            out_channels,
            kernel_size=kernel_size,
            padding=self.padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (batch, seq_len, 2) or (batch, 2)
        Returns:
            compensated tensor of same shape as input
        """
        if x.dim() == 2:
            # Single sample: (batch, 2) -> (batch, 2, 1)
            x = x.unsqueeze(-1)
            squeeze_out = True
        else:
            # Sequence: (batch, seq_len, 2) -> (batch, 2, seq_len)
            x = x.transpose(1, 2)
            squeeze_out = False

        # Initial conv: (batch, 2, seq_len) -> (batch, channels, seq_len)
        h = self.act_in(self.bn_in(self.conv_in(x)))

        # Residual blocks
        for block in self.blocks:
            residual = h
            h = block(h)
            if h.shape[1] != residual.shape[1]:
                residual = self._conv_match(residual, h.shape[1])
            h = h + residual

        # Output
        out = self.conv_out(h)

        if squeeze_out:
            return out.squeeze(-1).transpose(0, 1) if out.dim() == 3 else out
        else:
            return out.transpose(1, 2)

    def _conv_match(self, x: torch.Tensor, out_channels: int) -> torch.Tensor:
        """1x1 conv to match residual dimensions."""
        return nn.Conv1d(x.shape[1], out_channels, kernel_size=1, device=x.device)(x)

    def get_flops_per_symbol(self) -> float:
        """Estimate FLOPs per symbol."""
        total_params = sum(p.numel() for p in self.parameters())
        return float(total_params * 2)


class ResCNN_NLC(nn.Module):
    """
    Deeper residual CNN with dilated convolutions for larger context.

    Designed to capture long-range dispersion effects across hundreds of symbols.

    Args:
        in_channels: input channels (2 for I/Q)
        out_channels: output channels (2 for I/Q)
        num_layers: number of conv layers
        channels: number of channels per layer
        kernel_size: convolution kernel size
        dilation_base: base for dilation rate growth (2^i)
        dropout: dropout probability
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        num_layers: int = 8,
        channels: int = 64,
        kernel_size: int = 5,
        dilation_base: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, channels, kernel_size=1),
            nn.GELU(),
        )

        # Dilated residual layers
        for i in range(num_layers):
            dilation = dilation_base ** (i % 4)
            padding = (kernel_size - 1) * dilation // 2
            self.layers.append(nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=kernel_size,
                          padding=padding, dilation=dilation),
                nn.BatchNorm1d(channels),
                nn.GELU(),
                nn.Dropout(dropout),
            ))

        # Output projection
        self.output_proj = nn.Conv1d(channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input (batch, 2) or (batch, seq_len, 2)
        Returns:
            output (batch, 2) or (batch, seq_len, 2)
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            squeeze_out = True
        else:
            x = x.transpose(1, 2)
            squeeze_out = False

        h = self.input_proj(x)

        for layer in self.layers:
            h = h + layer(h)  # Residual connection

        out = self.output_proj(h)

        if squeeze_out:
            return out.squeeze(-1).transpose(0, 1) if out.dim() == 3 else out
        else:
            return out.transpose(1, 2)

    def get_flops_per_symbol(self) -> float:
        total_params = sum(p.numel() for p in self.parameters())
        return float(total_params * 2)
