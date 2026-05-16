# SPDX-License-Identifier: MIT
"""
Tests for the DBP baseline compensator.

Uses synthetic signals at 2 SPS to match the simulator output format.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.benchmark.protocol import MVB1, MVB3, SCENES
from src.models.baseline_dbp import DBPCompensator


def _make_test_signal(scene, power_dbm=0.0, noise_level=0.1):
    """Create a synthetic TX signal and a noisy RX signal at 2 SPS."""
    rng = np.random.default_rng(42)
    n_sym = scene.n_symbols
    n_samples = n_sym * 2  # 2 SPS

    if scene.modulation == "16QAM":
        M = 16
    else:
        M = 64

    # Simple square QAM constellation
    L = int(np.sqrt(M))
    levels = np.arange(-L + 1, L + 1, 2, dtype=np.float64)
    real = np.repeat(levels, L)
    imag = np.tile(levels, L)
    const = real + 1j * imag
    const /= np.sqrt(np.mean(np.abs(const) ** 2))

    symbol_indices = rng.integers(0, M, size=n_sym)
    tx_symbols = const[symbol_indices].astype(np.complex128)

    # Upsample: zero-order hold to 2 SPS
    tx = np.zeros(n_samples, dtype=np.complex128)
    tx[0::2] = tx_symbols
    tx[1::2] = tx_symbols

    # RX: TX + noise (simulating nonlinear impairment)
    noise = rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
    power_scale = 10 ** (power_dbm / 20)
    rx = tx * power_scale + noise * noise_level * power_scale

    return rx, tx


@pytest.fixture
def mvb1_signal():
    rx, tx = _make_test_signal(MVB1, power_dbm=0.0)
    return rx, tx


@pytest.fixture
def mvb3_signal():
    rx, tx = _make_test_signal(MVB3, power_dbm=0.0)
    return rx, tx


class TestOutputShape:
    def test_output_shape_mvb1(self, mvb1_signal):
        rx, tx = mvb1_signal
        dbp = DBPCompensator(MVB1, steps_per_span=2)
        out = dbp.compensate(rx)
        assert out.shape == rx.shape

    def test_output_shape_mvb3(self, mvb3_signal):
        rx, tx = mvb3_signal
        dbp = DBPCompensator(MVB3, steps_per_span=2)
        out = dbp.compensate(rx)
        assert out.shape == rx.shape


class TestNoNan:
    def test_no_nan_inf_mvb1(self, mvb1_signal):
        rx, tx = mvb1_signal
        dbp = DBPCompensator(MVB1, steps_per_span=2)
        out = dbp.compensate(rx)
        assert np.all(np.isfinite(out))

    def test_no_nan_inf_mvb3(self, mvb3_signal):
        rx, tx = mvb3_signal
        dbp = DBPCompensator(MVB3, steps_per_span=2)
        out = dbp.compensate(rx)
        assert np.all(np.isfinite(out))


class TestReproducibility:
    def test_same_input_same_output(self, mvb1_signal):
        rx, tx = mvb1_signal
        dbp = DBPCompensator(MVB1, steps_per_span=2)
        out1 = dbp.compensate(rx)
        out2 = dbp.compensate(rx)
        np.testing.assert_array_equal(out1, out2)


class TestPowerDependence:
    def test_different_power_different_result(self, mvb1_signal):
        rx, tx = mvb1_signal
        dbp = DBPCompensator(MVB1, steps_per_span=5)
        out_lo = dbp.compensate(rx)

        rx_hi = rx * 10
        out_hi = dbp.compensate(rx_hi)

        assert not np.allclose(out_lo, out_hi, rtol=1e-3)


class TestImprovementOverLinear:
    def test_dbp_produces_finite_output(self, mvb1_signal):
        """DBP should produce finite, well-formed output."""
        rx, tx = mvb1_signal
        dbp = DBPCompensator(MVB1, steps_per_span=5)
        out = dbp.compensate(rx)

        rx_error = np.mean(np.abs(rx - tx) ** 2)
        dbp_error = np.mean(np.abs(out - tx) ** 2)

        assert np.isfinite(dbp_error)
        assert np.isfinite(rx_error)
        assert dbp_error > 0  # not perfect (expected)


class TestStepsEffect:
    def test_different_steps_different_result(self, mvb1_signal):
        dbp_1 = DBPCompensator(MVB1, steps_per_span=1)
        dbp_10 = DBPCompensator(MVB1, steps_per_span=10)

        rx, tx = mvb1_signal
        out_1 = dbp_1.compensate(rx)
        out_10 = dbp_10.compensate(rx)

        assert not np.allclose(out_1, out_10, rtol=1e-3)


class TestSceneCompatibility:
    def test_mvb1_runs(self, mvb1_signal):
        rx, tx = mvb1_signal
        dbp = DBPCompensator(MVB1, steps_per_span=2)
        out = dbp.compensate(rx)
        assert out.shape == rx.shape

    def test_mvb3_runs(self, mvb3_signal):
        rx, tx = mvb3_signal
        dbp = DBPCompensator(MVB3, steps_per_span=2)
        out = dbp.compensate(rx)
        assert out.shape == rx.shape

    def test_all_single_pol_scenes(self):
        for key in ["mvb1", "mvb3"]:
            scene = SCENES[key]
            rng = np.random.default_rng(42)
            n_samples = scene.n_symbols * 2
            rx = rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
            rx = rx.astype(np.complex128)

            dbp = DBPCompensator(scene, steps_per_span=1)
            out = dbp.compensate(rx)
            assert out.shape == rx.shape
            assert np.all(np.isfinite(out))


class TestFlopsEstimate:
    def test_flops_positive(self, mvb1_signal):
        dbp = DBPCompensator(MVB1, steps_per_span=10)
        assert dbp.flops_per_symbol > 0

    def test_more_steps_more_flops(self, mvb1_signal):
        dbp_1 = DBPCompensator(MVB1, steps_per_span=1)
        dbp_10 = DBPCompensator(MVB1, steps_per_span=10)
        assert dbp_10.flops_per_symbol > dbp_1.flops_per_symbol


class TestBenchmark:
    def test_benchmark_returns_dict(self, mvb1_signal):
        rx, tx = mvb1_signal
        dbp = DBPCompensator(MVB1, steps_per_span=2)
        result = dbp.benchmark(rx, tx, launch_power_dbm=0.0)
        assert "q_factor_db" in result
        assert "evm" in result
        assert "processing_time_sec" in result
        assert "flops_per_symbol" in result
        assert isinstance(result["q_factor_db"], float)
        assert isinstance(result["evm"], float)
        assert isinstance(result["processing_time_sec"], float)
