# SPDX-License-Identifier: MIT
"""
Tests for the fiber simulator module.

OptiCommPy SSFM is mocked to keep tests fast.
Real QAM constellation is used (it is deterministic and fast).
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.benchmark.protocol import MVB1, MVB2, MVB3, SCENES, SceneParams
from src.data.simulator import FiberSimulator


# ──────────────────────────────────────────────
# Mock OptiCommPy SSFM (fast substitute)
# ──────────────────────────────────────────────


class _MockParam:
    """Minimal mock of optic.utils.parameters."""
    def __init__(self):
        self.Ltotal = 800.0
        self.Lspan = 80.0
        self.hz = 0.5
        self.alpha = 0.2
        self.D = 16.0
        self.gamma = 1.3
        self.Fc = 193.1e12
        self.Fs = 64e9
        self.prec = np.complex128
        self.amp = "edfa"
        self.NF = 4.5
        self.prgsBar = False


def _mock_ssfm(Ei, param):
    """Mock SSFM: return signal with same shape, slightly perturbed."""
    rng = np.random.default_rng(123)
    noise = (rng.standard_normal(Ei.shape) + 1j * rng.standard_normal(Ei.shape)) * 0.05
    return Ei + noise.astype(Ei.dtype)


def _mock_manakov(Ei, param):
    """Mock Manakov SSFM for dual-pol."""
    rng = np.random.default_rng(123)
    noise = (rng.standard_normal(Ei.shape) + 1j * rng.standard_normal(Ei.shape)) * 0.05
    return Ei + noise.astype(Ei.dtype)


def _mock_params_cls():
    return _MockParam()


@pytest.fixture
def mock_opticommpy():
    """Patch OptiCommPy: real QAM constellation, mocked SSFM."""
    from optic.comm.modulation import qamConst

    with patch("src.data.simulator._import_opticommpy") as mock_import:
        mock_import.return_value = (_mock_ssfm, _mock_manakov, qamConst, _mock_params_cls)
        yield mock_import


# ──────────────────────────────────────────────
# Init tests
# ──────────────────────────────────────────────


class TestFiberSimulatorInit:
    def test_mvb1_single_pol(self, mock_opticommpy):
        sim = FiberSimulator(MVB1)
        assert sim.dual_pol is False
        assert sim.M == 16
        assert sim.n_symbols == 2 ** 16
        assert sim.Fs == 32e9 * sim.sps

    def test_mvb2_dual_pol(self, mock_opticommpy):
        sim = FiberSimulator(MVB2)
        assert sim.dual_pol is True
        assert sim.M == 16

    def test_mvb3_64qam(self, mock_opticommpy):
        sim = FiberSimulator(MVB3)
        assert sim.M == 64
        assert sim.dual_pol is False


# ──────────────────────────────────────────────
# Output shape tests
# ──────────────────────────────────────────────


class TestOutputShape:
    def test_single_pol_shape(self, mock_opticommpy):
        sim = FiberSimulator(MVB1)
        rx, tx = sim.propagate(0.0)
        assert rx.shape == (MVB1.n_symbols,)
        assert tx.shape == (MVB1.n_symbols,)
        assert rx.dtype == np.complex128

    def test_dual_pol_shape(self, mock_opticommpy):
        sim = FiberSimulator(MVB2)
        rx, tx = sim.propagate(0.0)
        assert rx.shape == (2, MVB2.n_symbols)
        assert tx.shape == (2, MVB2.n_symbols)


# ──────────────────────────────────────────────
# Reproducibility tests
# ──────────────────────────────────────────────


class TestReproducibility:
    def test_same_seed_same_output(self, mock_opticommpy):
        sim1 = FiberSimulator(MVB1)
        sim2 = FiberSimulator(MVB1)
        rx1, tx1 = sim1.propagate(1.0)
        rx2, tx2 = sim2.propagate(1.0)
        np.testing.assert_array_equal(rx1, rx2)
        np.testing.assert_array_equal(tx1, tx2)

    def test_different_seed_different_tx(self, mock_opticommpy):
        sim_a = FiberSimulator(MVB1)
        scene_b = SceneParams(name="MVB-1-alt", seed=99, modulation="16QAM", polarization="single")
        sim_b = FiberSimulator(scene_b)
        _, tx_a = sim_a.propagate(1.0)
        _, tx_b = sim_b.propagate(1.0)
        assert not np.array_equal(tx_a, tx_b)


# ──────────────────────────────────────────────
# Power sweep tests
# ──────────────────────────────────────────────


class TestPowerSweep:
    def test_power_sweep_count(self, mock_opticommpy):
        sim = FiberSimulator(MVB1)
        results = sim.run_power_sweep(progress=False)
        assert len(results) == MVB1.tx_power_points

    def test_power_sweep_range(self, mock_opticommpy):
        sim = FiberSimulator(MVB1)
        results = sim.run_power_sweep(progress=False)
        powers = list(results.keys())
        lo, hi = MVB1.tx_power_range_dbm
        assert powers[0] == pytest.approx(lo)
        assert powers[-1] == pytest.approx(hi)

    def test_power_sweep_uniform_spacing(self, mock_opticommpy):
        sim = FiberSimulator(MVB1)
        results = sim.run_power_sweep(progress=False)
        powers = list(results.keys())
        diffs = [powers[i + 1] - powers[i] for i in range(len(powers) - 1)]
        expected_step = (MVB1.tx_power_range_dbm[1] - MVB1.tx_power_range_dbm[0]) / (MVB1.tx_power_points - 1)
        for d in diffs:
            assert d == pytest.approx(expected_step)


# ──────────────────────────────────────────────
# Scene params integration
# ──────────────────────────────────────────────


class TestSceneParamsIntegration:
    def test_all_scenes_run(self, mock_opticommpy):
        for key, scene in SCENES.items():
            sim = FiberSimulator(scene)
            rx, tx = sim.propagate(0.0)
            assert rx is not None
            assert tx is not None


# ──────────────────────────────────────────────
# Power scaling
# ──────────────────────────────────────────────


class TestPowerScaling:
    def test_high_power_higher_amplitude(self, mock_opticommpy):
        sim = FiberSimulator(MVB1)
        _, tx_lo = sim.propagate(-3.0)
        _, tx_hi = sim.propagate(5.0)
        assert np.abs(tx_hi).mean() > np.abs(tx_lo).mean()
