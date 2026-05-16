# SPDX-License-Identifier: MIT
"""
MLP-based Nonlinear Compensation model.

A feedforward neural network that learns the inverse mapping of the fiber
channel, compensating for nonlinear distortion on a per-sample basis.

Input: complex optical field → split into I/Q channels (2D real)
Output: compensated complex optical field (2D real)

This is a memoryless model (no temporal context). For dispersion-aware
compensation, see CNN-NLC or Transformer-NLC (Phase 2+).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class MLP_NLC(nn.Module):
    """
    Multi-Layer Perceptron for nonlinear compensation.

    Maps impaired complex samples to compensated ones by learning the
    inverse fiber channel mapping.

    Args:
        input_dim: input feature dimension (default 2 for I/Q)
        hidden_dims: list of hidden layer sizes
        output_dim: output feature dimension (default 2 for I/Q)
        dropout: dropout probability (0 = disabled)
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: list[int] = [128, 128, 64],
        output_dim: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (N, 2) for I/Q components
        Returns:
            compensated tensor of shape (N, 2)
        """
        return self.network(x)


class MLPWithMemory_NLC(nn.Module):
    """
    MLP with temporal context window for dispersion-aware compensation.

    Chromatic dispersion causes inter-symbol interference, so a memoryless
    model cannot fully compensate. This model uses a sliding window of
    surrounding samples as input context.

    Args:
        memory_size: number of surrounding samples on each side
                     (total window = 2 * memory_size + 1)
        hidden_dims: list of hidden layer sizes
        dropout: dropout probability
    """

    def __init__(
        self,
        memory_size: int = 5,
        hidden_dims: list[int] = [256, 256, 128],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.memory_size = memory_size
        # Input dimension: 2 (I/Q) * (2*memory + 1) samples
        self.input_dim = 2 * (2 * memory_size + 1)

        layers: list[nn.Module] = []
        prev_dim = self.input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 2))  # Output: compensated I/Q

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (N, 2*memory+1, 2)
               where the middle dimension is the context window
        Returns:
            compensated tensor of shape (N, 2) — the center sample
        """
        # Flatten context window: (N, window, 2) → (N, window*2)
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        return self.network(x_flat)

    def get_flops_per_symbol(self) -> float:
        """
        Estimate FLOPs per symbol for this model.
        Approximate: 2 * weights for forward pass.
        """
        total_params = sum(p.numel() for p in self.parameters())
        # Rough estimate: 2 FLOPs per parameter (multiply + add)
        return float(total_params * 2)
