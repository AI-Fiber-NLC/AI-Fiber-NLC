# SPDX-License-Identifier: MIT
"""
Fiber simulation data generator for AI-Fiber-NLC.

Uses OptiCommPy to simulate nonlinear fiber propagation and generate
(impaired, clean) signal pairs for training NLC models.
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


class FiberSimulator:
    """
    Wraps OptiCommPy SSFM simulation for NLC data generation.

    Each instance is bound to a single SceneParams configuration.
    Generates data at 2 samples per symbol (SPS) for proper SSFM operation.
    """

    def __init__(self, scene: SceneParams, sps: int = 2):
        self.scene = scene
        self.sps = sps
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

    def _upsample(self, symbols: np.ndarray) -> np.ndarray:
        """Map symbols to constellation and upsample to 2 SPS."""
        const = self._qam_const(self.M).flatten()

        if self.dual_pol:
            field = np.zeros((2, self.n_samples), dtype=np.complex128)
            field[0, ::self.sps] = const[symbols[0]]
            field[1, ::self.sps] = const[symbols[1]]
            # Zero-order hold for intermediate samples
            for pol in range(2):
                for s in range(1, self.sps):
                    field[pol, s::self.sps] = field[pol, 0::self.sps]
            return field
        else:
            field = np.zeros(self.n_samples, dtype=np.complex128)
            field[::self.sps] = const[symbols]
            for s in range(1, self.sps):
                field[s::self.sps] = field[0::self.sps]
            return field

    def _scale_to_power(self, field: np.ndarray, power_dbm: float) -> np.ndarray:
        target_W = 10 ** (power_dbm / 10) / 1000.0
        scale = np.sqrt(target_W)
        if self.dual_pol:
            scale /= np.sqrt(2)
        return field * scale

    def propagate(self, tx_power_dbm: float, progress: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        np.random.seed(self.scene.seed)
        symbols = self._generate_symbols()
        tx_field = self._upsample(symbols)
        tx_field = self._scale_to_power(tx_field, tx_power_dbm)
        param = self._build_param(prgs_bar=progress)

        if self.dual_pol:
            rx_field = self._manakov(tx_field, param)
        else:
            rx_field = self._ssfm(tx_field, param)

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
