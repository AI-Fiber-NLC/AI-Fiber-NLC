# SPDX-License-Identifier: MIT
"""
Fiber simulation data generator for AI-Fiber-NLC.

Uses OptiCommPy to simulate nonlinear fiber propagation and generate
(impaired, clean) signal pairs for training NLC models.

Signal format: 2 samples per symbol (2 SPS), generated with RRC pulse
shaping for realistic band-limited transmission.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple

from src.benchmark.protocol import SceneParams, SCENES


def _import_opticommpy():
    """Import OptiCommPy modules on demand."""
    from optic.models.channels import ssfm, manakovSSF
    from optic.comm.modulation import qamConst
    from optic.utils import parameters
    return ssfm, manakovSSF, qamConst, parameters


def rrc_pulse_shape(symbols: np.ndarray,
                    sps: int = 2,
                    alpha: float = 0.01,
                    span: int = 10) -> np.ndarray:
    """
    Apply Root-Raised Cosine (RRC) pulse shaping to QAM symbols.

    Args:
        symbols: QAM symbol sequence (complex, 1D)
        sps: samples per symbol
        alpha: roll-off factor (default 0.01 for near-Nyquist)
        span: filter span in symbols

    Returns:
        Pulse-shaped signal at sps samples per symbol
    """
    from optic.dsp.core import rrcFilterTaps, firFilter

    n_symbols = len(symbols)
    n_taps = span * sps + 1

    # Generate RRC filter taps
    t = np.arange(-span * sps // 2, span * sps // 2 + 1) / sps
    Ts = 1.0
    taps = rrcFilterTaps(t, alpha, Ts)

    # Upsample: insert (sps-1) zeros between symbols
    upsampled = np.zeros(n_symbols * sps, dtype=np.complex128)
    upsampled[::sps] = symbols

    # Apply RRC filter (firFilter compensates for delay)
    shaped = firFilter(taps, upsampled)

    # firFilter may change length; trim or pad to target
    target_len = n_symbols * sps
    if len(shaped) > target_len:
        return shaped[:target_len]
    elif len(shaped) < target_len:
        return np.pad(shaped, (0, target_len - len(shaped)), mode='constant')
    return shaped


class FiberSimulator:
    """
    Wraps OptiCommPy SSFM simulation for NLC data generation.

    Generates data with RRC pulse shaping at 2 samples per symbol.
    """

    def __init__(self, scene: SceneParams, sps: int = 2, rrc_alpha: float = 0.01):
        self.scene = scene
        self.sps = sps
        self.rrc_alpha = rrc_alpha
        self.Fs = scene.baud_rate_GBd * 1e9 * sps
        self.Fc = 193.1e12
        self.n_symbols = scene.n_symbols
        self.n_samples = self.n_symbols * sps
        self.dual_pol = scene.polarization == "dual"

        if scene.modulation == "16QAM":
            self.M = 16
        elif scene.modulation == "64QAM":
            self.M = 64
        else:
            raise ValueError(f"Unsupported modulation: {scene.modulation}")

        ssfm_fn, manakov_fn, qam_fn, param_cls = _import_opticommpy()
        self._ssfm = ssfm_fn
        self._manakov = manakov_fn
        self._qam_const = qam_fn
        self._param_cls = param_cls

    def _build_param(self, prgs_bar: bool = False) -> object:
        p = self._param_cls()
        p.Ltotal = self.scene.fiber_length_km * self.scene.num_spans
        p.Lspan = self.scene.fiber_length_km
        p.hz = 0.5
        p.alpha = self.scene.alpha_db_per_km
        p.D = self.scene.D_ps_per_nm_km
        p.gamma = self.scene.gamma_per_W_km
        p.Fc = self.Fc
        p.Fs = self.Fs
        p.prec = np.complex128
        p.amp = "edfa"
        p.NF = 4.5
        p.prgsBar = prgs_bar
        return p

    def _generate_symbols(self) -> np.ndarray:
        rng = np.random.default_rng(self.scene.seed)
        if self.dual_pol:
            syms_x = rng.integers(0, self.M, size=self.n_symbols)
            syms_y = rng.integers(0, self.M, size=self.n_symbols)
            return np.stack([syms_x, syms_y])
        else:
            return rng.integers(0, self.M, size=self.n_symbols)

    def _pulse_shape(self, symbol_indices: np.ndarray) -> np.ndarray:
        """Map symbols to constellation and apply RRC pulse shaping."""
        const = self._qam_const(self.M).flatten()
        symbols = const[symbol_indices].astype(np.complex128)

        if self.dual_pol:
            field_x = rrc_pulse_shape(symbols[0], self.sps, self.rrc_alpha)
            field_y = rrc_pulse_shape(symbols[1], self.sps, self.rrc_alpha)
            # Ensure same length
            min_len = min(len(field_x), len(field_y))
            return np.stack([field_x[:min_len], field_y[:min_len]])
        else:
            return rrc_pulse_shape(symbols, self.sps, self.rrc_alpha)

    def _scale_to_power(self, field: np.ndarray, power_dbm: float) -> np.ndarray:
        """Scale the optical field to the target launch power."""
        if self.dual_pol:
            current_power = np.mean(np.abs(field[0]) ** 2) + np.mean(np.abs(field[1]) ** 2)
        else:
            current_power = np.mean(np.abs(field) ** 2)

        if current_power <= 0:
            return field

        target_W = 10 ** (power_dbm / 10) / 1000.0
        scale = np.sqrt(target_W / current_power)

        if self.dual_pol:
            scale /= np.sqrt(2)

        return field * scale

    def propagate(self, tx_power_dbm: float, progress: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run a single fiber propagation simulation.

        Returns:
            (rx_field, tx_field) — both as complex numpy arrays at 2 SPS
            with RRC pulse shaping.
        """
        np.random.seed(self.scene.seed)

        symbols = self._generate_symbols()
        tx_field = self._pulse_shape(symbols)
        tx_field = self._scale_to_power(tx_field, tx_power_dbm)

        param = self._build_param(prgs_bar=progress)

        if self.dual_pol:
            rx_field = self._manakov(tx_field, param)
        else:
            rx_field = self._ssfm(tx_field, param)

        # Ensure rx and tx have the same length
        if self.dual_pol:
            min_len = min(rx_field.shape[1], tx_field.shape[1])
            rx_field = rx_field[:, :min_len]
            tx_field = tx_field[:, :min_len]
        else:
            min_len = min(len(rx_field), len(tx_field))
            rx_field = rx_field[:min_len]
            tx_field = tx_field[:min_len]

        return rx_field, tx_field

    def run_power_sweep(self, progress: bool = True) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
        lo, hi = self.scene.tx_power_range_dbm
        n_pts = self.scene.tx_power_points
        powers = np.linspace(lo, hi, n_pts)
        results: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}

        for i, pwr in enumerate(powers):
            print(f"  [{i + 1}/{n_pts}] Power: {pwr:+.1f} dBm ...", end=" ", flush=True)
            rx, tx = self.propagate(pwr, progress=progress)
            results[float(pwr)] = (rx, tx)
            print(f"done (rx shape: {rx.shape})")

        return results
