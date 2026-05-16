# SPDX-License-Identifier: MIT
"""
Digital Back Propagation (DBP) baseline compensator.

Implements symmetric split-step Fourier method for single-polarization
nonlinear compensation. Uses OptiCommPy's SSFM engine with reversed
dispersion and nonlinearity coefficients to undo fiber impairments.

After DBP, the signal is normalized to match the TX signal power level
(since EDFA in the forward simulation and DBP may produce different
absolute power levels).

Reference:
    E. Ip and J. M. Kahn, "Compensation of dispersion and nonlinear
    impairments using digital backpropagation", JLT, 26(20), 2008.
"""

from __future__ import annotations

import numpy as np
import time
import math

from src.benchmark.protocol import SceneParams


class DBPCompensator:
    """
    Digital Back Propagation compensator for single-polarization signals.

    Uses OptiCommPy's SSFM with negated gamma and D for backward propagation.
    Output is power-normalized to match the expected TX signal level.

    Example:
        >>> from src.benchmark.protocol import MVB1
        >>> dbp = DBPCompensator(MVB1, steps_per_span=10)
        >>> compensated = dbp.compensate(rx_signal)
    """

    def __init__(self, scene: SceneParams, steps_per_span: int = 10):
        self.scene = scene
        self.steps_per_span = steps_per_span
        self.n_spans = scene.num_spans
        self.span_length = scene.fiber_length_km
        self.sps = 2  # matches simulator output
        self.n_symbols = scene.n_symbols
        self.n_samples = self.n_symbols * self.sps
        self.Fs = scene.baud_rate_GBd * 1e9 * self.sps

        # FLOPs estimate
        log_n = math.log2(max(self.n_samples, 2))
        flops_per_step = 10.0 * self.n_samples * log_n + 20.0 * self.n_samples
        total_steps = self.n_spans * self.steps_per_span
        self.flops_per_symbol = total_steps * flops_per_step / self.n_symbols

    def compensate(self,
                   rx_signal: np.ndarray,
                   launch_power_dbm: float = 0.0) -> np.ndarray:
        """
        Apply DBP to a received signal.

        Uses OptiCommPy SSFM with negated gamma and D.
        Output is power-normalized to match the input signal power.

        Args:
            rx_signal: received optical field, shape (n_samples,)
            launch_power_dbm: launch power used during forward propagation

        Returns:
            compensated_signal, same shape as input, power-normalized
        """
        from optic.models.channels import ssfm
        from optic.utils import parameters

        if rx_signal.shape != (self.n_samples,):
            raise ValueError(
                f"Expected signal shape ({self.n_samples},), "
                f"got {rx_signal.shape}"
            )

        input_power = np.mean(np.abs(rx_signal) ** 2)

        p = parameters()
        p.Ltotal = self.scene.fiber_length_km * self.scene.num_spans
        p.Lspan = self.scene.fiber_length_km
        p.hz = self.scene.fiber_length_km / self.steps_per_span
        p.alpha = self.scene.alpha_db_per_km
        # Negate D to reverse dispersion
        p.D = -self.scene.D_ps_per_nm_km
        # Negate gamma to reverse nonlinearity
        p.gamma = -self.scene.gamma_per_W_km
        p.Fc = 193.1e12
        p.Fs = self.Fs
        p.prec = np.complex128
        # Use no amplifier (amp=None) to avoid EDFA noise and ensure reproducibility.
        # Power normalization at the end handles amplitude changes.
        p.amp = "None"
        p.prgsBar = False

        result = ssfm(rx_signal.astype(np.complex128), p)

        # Power-normalize to match input signal level
        output_power = np.mean(np.abs(result) ** 2)
        if output_power > 0:
            scale = np.sqrt(input_power / output_power)
            result = result * scale

        return result

    def benchmark(self,
                  rx_signal: np.ndarray,
                  tx_signal: np.ndarray,
                  launch_power_dbm: float = 0.0) -> dict:
        """Run DBP and compute performance metrics."""
        t0 = time.perf_counter()
        compensated = self.compensate(rx_signal, launch_power_dbm)
        elapsed = time.perf_counter() - t0

        error = compensated - tx_signal
        signal_power = np.mean(np.abs(tx_signal) ** 2)
        error_power = np.mean(np.abs(error) ** 2)

        if error_power > 0 and signal_power > 0:
            evm = np.sqrt(error_power / signal_power)
            q_factor = 20 * np.log10(1.0 / evm) if evm > 0 else float('inf')
        else:
            evm = 0.0
            q_factor = 99.9

        return {
            "q_factor_db": round(q_factor, 3) if np.isfinite(q_factor) else 99.9,
            "evm": round(evm, 6),
            "processing_time_sec": round(elapsed, 3),
            "flops_per_symbol": round(self.flops_per_symbol, 1),
        }
