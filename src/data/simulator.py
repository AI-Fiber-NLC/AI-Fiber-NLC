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
    """

    def __init__(self, scene: SceneParams, sps: int = 2):
        """
        Args:
            scene: benchmark scene definition (from protocol.py).
            sps: samples per symbol for the SSFM simulation.
        """
        self.scene = scene
        self.sps = sps

        # Sampling frequency [Hz] = baud_rate * samples_per_symbol
        self.Fs = scene.baud_rate_GBd * 1e9 * sps

        # Carrier frequency [Hz] — C-band center
        self.Fc = 193.1e12

        # Number of symbols
        self.n_symbols = scene.n_symbols

        # Dual-polarization flag
        self.dual_pol = scene.polarization == "dual"

        # QAM order
        if scene.modulation == "16QAM":
            self.M = 16
        elif scene.modulation == "64QAM":
            self.M = 64
        else:
            raise ValueError(f"Unsupported modulation: {scene.modulation}")

        # Build OptiCommPy parameter object
        ssfm_fn, manakov_fn, qam_fn, param_cls = _import_opticommpy()
        self._ssfm = ssfm_fn
        self._manakov = manakov_fn
        self._qam_const = qam_fn
        self._param_cls = param_cls

    def _build_param(self, prgs_bar: bool = False) -> object:
        """Build OptiCommPy parameters object from SceneParams."""
        p = self._param_cls()
        p.Ltotal = self.scene.fiber_length_km * self.scene.num_spans
        p.Lspan = self.scene.fiber_length_km
        p.hz = 0.5                    # SSFM step size [km]
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
        """
        Generate random QAM symbol indices.

        Returns:
            For single-pol: 1D array of shape (n_symbols,)
            For dual-pol:   2D array of shape (2, n_symbols)
        """
        rng = np.random.default_rng(self.scene.seed)

        if self.dual_pol:
            syms_x = rng.integers(0, self.M, size=self.n_symbols)
            syms_y = rng.integers(0, self.M, size=self.n_symbols)
            return np.stack([syms_x, syms_y])
        else:
            return rng.integers(0, self.M, size=self.n_symbols)

    def _map_to_field(self, symbols: np.ndarray) -> np.ndarray:
        """
        Map symbol indices to QAM constellation points.

        The constellation is normalized to unit average power.

        Args:
            symbols: output of _generate_symbols
        Returns:
            Complex optical field array.
        """
        const = self._qam_const(self.M).flatten()

        if self.dual_pol:
            field = np.zeros((2, len(symbols[0])), dtype=np.complex128)
            field[0] = const[symbols[0]]
            field[1] = const[symbols[1]]
            return field
        else:
            return const[symbols].astype(np.complex128)

    def _scale_to_power(self, field: np.ndarray, power_dbm: float) -> np.ndarray:
        """
        Scale the optical field to the target launch power.

        For a normalized QAM constellation (avg power = 1):
            target_linear_W = 10^(power_dbm / 10) / 1000
            scale = sqrt(target_linear_W)
        """
        target_W = 10 ** (power_dbm / 10) / 1000.0
        scale = np.sqrt(target_W)

        if self.dual_pol:
            # Each polarization gets half the power
            scale /= np.sqrt(2)

        return field * scale

    def propagate(self, tx_power_dbm: float, progress: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run a single fiber propagation simulation.

        Args:
            tx_power_dbm: launch power in dBm
            progress: whether to show OptiCommPy progress bar

        Returns:
            (rx_field, tx_field) — both as complex numpy arrays
            tx_field: transmitted constellation-mapped symbols (reference)
            rx_field: received signal after nonlinear fiber propagation
        """
        np.random.seed(self.scene.seed)

        # Generate and map symbols
        symbols = self._generate_symbols()
        tx_field = self._map_to_field(symbols)

        # Scale to target power
        tx_field = self._scale_to_power(tx_field, tx_power_dbm)

        # Build parameters
        param = self._build_param(prgs_bar=progress)

        # Propagate through fiber
        # OptiCommPy ssfm/manakovSSF return just the field array (not a tuple)
        if self.dual_pol:
            rx_field = self._manakov(tx_field, param)
        else:
            rx_field = self._ssfm(tx_field, param)

        return rx_field, tx_field

    def run_power_sweep(self, progress: bool = True) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
        """
        Run simulations for all power points in the scene.

        Returns:
            {tx_power_dbm: (rx_field, tx_field)}
        """
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
