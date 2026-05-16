# SPDX-License-Identifier: MIT
"""Tests for MLP-NLC models."""

from __future__ import annotations

import torch
import pytest
import numpy as np

from src.models.mlp_nlc import MLP_NLC, MLPWithMemory_NLC


class TestMLP_NLC:
    def test_output_shape(self):
        model = MLP_NLC(input_dim=2, hidden_dims=[32, 32], output_dim=2)
        x = torch.randn(100, 2)
        out = model(x)
        assert out.shape == (100, 2)

    def test_no_nan(self):
        model = MLP_NLC(input_dim=2, hidden_dims=[32, 32], output_dim=2)
        x = torch.randn(100, 2)
        out = model(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_gradient_flow(self):
        model = MLP_NLC(input_dim=2, hidden_dims=[32, 32], output_dim=2)
        x = torch.randn(50, 2)
        target = torch.randn(50, 2)
        criterion = torch.nn.MSELoss()
        out = model(x)
        loss = criterion(out, target)
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_dropout(self):
        model = MLP_NLC(input_dim=2, hidden_dims=[32, 32], output_dim=2, dropout=0.5)
        model.train()
        x = torch.randn(100, 2)
        out1 = model(x)
        out2 = model(x)
        # With dropout, outputs should differ
        assert not torch.allclose(out1, out2, atol=1e-6)

    def test_deterministic_without_dropout(self):
        model = MLP_NLC(input_dim=2, hidden_dims=[32, 32], output_dim=2, dropout=0.0)
        model.eval()
        x = torch.randn(100, 2)
        out1 = model(x)
        out2 = model(x)
        torch.testing.assert_close(out1, out2)


class TestMLPWithMemory_NLC:
    def test_output_shape(self):
        memory_size = 5
        model = MLPWithMemory_NLC(memory_size=memory_size, hidden_dims=[64, 64])
        # Input: (N, 2*memory+1, 2)
        window = 2 * memory_size + 1  # 11
        x = torch.randn(100, window, 2)
        out = model(x)
        assert out.shape == (100, 2)

    def test_no_nan(self):
        model = MLPWithMemory_NLC(memory_size=3, hidden_dims=[64, 64])
        window = 2 * 3 + 1  # 7
        x = torch.randn(100, window, 2)
        out = model(x)
        assert not torch.isnan(out).any()

    def test_gradient_flow(self):
        model = MLPWithMemory_NLC(memory_size=3, hidden_dims=[64, 64])
        window = 2 * 3 + 1
        x = torch.randn(50, window, 2)
        target = torch.randn(50, 2)
        criterion = torch.nn.MSELoss()
        out = model(x)
        loss = criterion(out, target)
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_memory_size_effect(self):
        model_s = MLPWithMemory_NLC(memory_size=1, hidden_dims=[64])
        model_l = MLPWithMemory_NLC(memory_size=10, hidden_dims=[64])
        assert model_l.input_dim > model_s.input_dim

    def test_flops_estimate(self):
        model = MLPWithMemory_NLC(memory_size=5, hidden_dims=[128, 128])
        flops = model.get_flops_per_symbol()
        assert flops > 0
