# SPDX-License-Identifier: MIT
"""
Tests for the NLC PyTorch Dataset.
"""

from __future__ import annotations

import numpy as np
import torch
import pytest
from pathlib import Path
import tempfile
import shutil

from src.data.dataset import NLCDataset


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory with synthetic .npz data."""
    tmp = Path(tempfile.mkdtemp())
    scene_dir = tmp / "MVB-1"
    scene_dir.mkdir()

    # Create 3 power-point files
    for pwr in [-3.0, 1.0, 5.0]:
        n_symbols = 256  # small for testing
        rng = np.random.default_rng(42)
        rx = rng.standard_normal(n_symbols) + 1j * rng.standard_normal(n_symbols)
        tx = rng.standard_normal(n_symbols) + 1j * rng.standard_normal(n_symbols)

        fname = f"p_{pwr:.1f}dBm.npz"
        np.savez_compressed(
            scene_dir / fname,
            rx_signal=rx.astype(np.complex64),
            tx_signal=tx.astype(np.complex64),
            tx_power_dbm=np.float64(pwr),
            scene_name=np.str_("MVB-1"),
            seed=np.int64(42),
        )

    yield tmp
    shutil.rmtree(tmp)


class TestDatasetLength:
    def test_length_matches_symbols(self, temp_data_dir):
        ds = NLCDataset(temp_data_dir, "MVB-1", tx_power_dbm=1.0)
        assert len(ds) == 256

    def test_length_all_power_points(self, temp_data_dir):
        for pwr in [-3.0, 1.0, 5.0]:
            ds = NLCDataset(temp_data_dir, "MVB-1", tx_power_dbm=pwr)
            assert len(ds) == 256


class TestLazyLoading:
    def test_no_load_on_init(self, temp_data_dir):
        ds = NLCDataset(temp_data_dir, "MVB-1", tx_power_dbm=1.0)
        assert ds._rx is None
        assert ds._tx is None

    def test_load_on_first_access(self, temp_data_dir):
        ds = NLCDataset(temp_data_dir, "MVB-1", tx_power_dbm=1.0)
        _ = len(ds)
        assert ds._rx is not None
        assert ds._tx is not None


class TestGetItemShape:
    def test_single_item_returns_iq_tensor(self, temp_data_dir):
        ds = NLCDataset(temp_data_dir, "MVB-1", tx_power_dbm=1.0)
        rx, tx = ds[0]
        # Complex → view_as_real → shape (2,) for I and Q
        assert rx.shape == (2,)
        assert tx.shape == (2,)
        assert isinstance(rx, torch.Tensor)
        assert isinstance(tx, torch.Tensor)

    def test_item_values_are_finite(self, temp_data_dir):
        ds = NLCDataset(temp_data_dir, "MVB-1", tx_power_dbm=1.0)
        rx, tx = ds[0]
        assert torch.isfinite(rx).all()
        assert torch.isfinite(tx).all()

    def test_all_indices_valid(self, temp_data_dir):
        ds = NLCDataset(temp_data_dir, "MVB-1", tx_power_dbm=1.0)
        for i in range(len(ds)):
            rx, tx = ds[i]
            assert rx.shape == (2,)
            assert tx.shape == (2,)


class TestListPowerFiles:
    def test_discovers_all_files(self, temp_data_dir):
        powers = NLCDataset.list_power_files(temp_data_dir, "MVB-1")
        assert powers == [-3.0, 1.0, 5.0]

    def test_empty_for_missing_scene(self, temp_data_dir):
        powers = NLCDataset.list_power_files(temp_data_dir, "MVB-99")
        assert powers == []


class TestMetadata:
    def test_metadata_accessible(self, temp_data_dir):
        ds = NLCDataset(temp_data_dir, "MVB-1", tx_power_dbm=1.0)
        meta = ds.metadata
        assert "tx_power_dbm" in meta
        assert "scene_name" in meta
        assert "seed" in meta
