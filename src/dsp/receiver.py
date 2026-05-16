# SPDX-License-Identifier: MIT
"""
DSP receiver chain for coherent optical communication.

Implements a complete digital signal processing pipeline:
1. Electronic Dispersion Compensation (EDC)
2. Clock recovery (Gardner algorithm, 2 SPS -> 1 SPS)
3. Carrier phase recovery (Viterbi-Viterbi for M-QAM)
4. Symbol detection and EVM-based Q-factor computation

This module provides the foundation for fair, reproducible benchmark
comparison across all NLC algorithms.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from src.benchmark.protocol import SceneParams


@dataclass
class DSPResult:
    """Result of DSP receiver chain processing."""
    compensated: np.ndarray    # Compensated signal at 1 SPS (complex)
    decided_symbols: np.ndarray  # Hard-decision constellation points (complex)
    evm: float                 # Error Vector Magnitude
    q_factor_db: float         # Q-factor in dB


def edc_compensate(signal: np.ndarray,
                   scene: SceneParams,
                   fs_hz: float) -> np.ndarray:
    """
    Electronic Dispersion Compensation.

    Removes chromatic dispersion accumulated over the fiber link using
    frequency-domain equalization (inverse of CD transfer function).

    Args:
        signal: received optical field (complex, at sampling rate fs_hz)
        scene: benchmark scene with fiber parameters
        fs_hz: sampling frequency in Hz

    Returns:
        Dispersion-compensated signal (same shape as input)
    """
    from optic.dsp.equalization import edc
    from optic.utils import parameters

    p = parameters()
    p.L = scene.fiber_length_km * scene.num_spans
    p.D = scene.D_ps_per_nm_km
    p.Fc = 193.1e12
    p.Fs = fs_hz
    p.Rs = scene.baud_rate_GBd * 1e9

    return edc(signal, p)


def clock_recovery(signal: np.ndarray) -> np.ndarray:
    """
    Clock recovery using Gardner's algorithm.

    Downsamples the signal from 2 SPS to 1 SPS, finding the optimal
    sampling instant.

    Args:
        signal: input signal at 2 SPS (complex)

    Returns:
        Signal at ~1 SPS (complex), length approx len(signal) / 2
    """
    from optic.dsp.clockRecovery import gardnerClockRecovery
    from optic.utils import parameters

    p = parameters()
    p.kp = 1e-3
    p.ki = 1e-6
    p.isNyquist = True
    p.returnTiming = False
    p.lpad = 1
    p.maxPPM = 500

    return gardnerClockRecovery(signal, p)


def carrier_phase_recovery(signal: np.ndarray,
                           m_qam: int = 16,
                           window_size: int = 51) -> np.ndarray:
    """
    Carrier phase recovery using Viterbi-Viterbi algorithm.

    Removes the carrier phase offset by raising the signal to the M-th
    power and averaging over a sliding window.

    Args:
        signal: input signal at 1 SPS (complex)
        m_qam: QAM order (16 for 16QAM, 64 for 64QAM)
        window_size: moving average window size

    Returns:
        Phase-corrected signal at 1 SPS (complex)
    """
    from optic.dsp.carrierRecovery import viterbi

    m_power = 4

    signal_2d = signal[np.newaxis, :]
    result = viterbi(signal_2d, N=window_size, M=m_power)

    if hasattr(result, 'ndim') and result.ndim == 2:
        return result[0]
    elif isinstance(result, tuple):
        return result[0][0] if hasattr(result[0], 'ndim') and result[0].ndim == 2 else result[0]
    else:
        return result


def symbol_detect(signal: np.ndarray,
                  m_qam: int = 16) -> Tuple[np.ndarray, float]:
    """
    Make hard symbol decisions and compute EVM.

    Uses nearest-neighbor decision on the QAM constellation.

    Args:
        signal: phase-corrected signal at 1 SPS (complex)
        m_qam: QAM order

    Returns:
        (decided_symbols, evm)
    """
    from optic.comm.modulation import qamConst

    const = qamConst(m_qam).flatten()

    n = len(signal)
    chunk = 10000
    all_decided = []

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        chunk_sig = signal[start:end]
        distances = np.abs(const[np.newaxis, :] - chunk_sig[:, np.newaxis])
        idx = np.argmin(distances, axis=1)
        all_decided.append(const[idx])

    decided = np.concatenate(all_decided)
    error = signal[:len(decided)] - decided

    signal_power = np.mean(np.abs(decided) ** 2)
    error_power = np.mean(np.abs(error) ** 2)

    if signal_power > 0:
        evm = np.sqrt(error_power / signal_power)
    else:
        evm = 1.0

    return decided, evm


def compute_q_factor_from_evm(evm: float) -> float:
    """
    Compute Q-factor (dB) from EVM.

    Q = 20 * log10(1 / EVM)
    """
    if evm <= 0:
        return 99.9
    return 20.0 * np.log10(1.0 / evm)


def process_receiver(rx_signal: np.ndarray,
                     scene: SceneParams,
                     fs_hz: float,
                     m_qam: int = 16,
                     cpr_window: int = 51,
                     skip_edc: bool = False) -> DSPResult:
    """
    Complete DSP receiver chain.

    Args:
        rx_signal: received optical field (complex, at fs_hz sampling rate)
        scene: benchmark scene
        fs_hz: sampling frequency in Hz
        m_qam: QAM constellation order
        cpr_window: carrier phase recovery window size
        skip_edc: if True, skip EDC (for DBP-compensated signals where
                  chromatic dispersion is already compensated)

    Returns:
        DSPResult with compensated signal and quality metrics
    """
    if skip_edc:
        processed_signal = rx_signal
    else:
        processed_signal = edc_compensate(rx_signal, scene, fs_hz)

    cr_signal = clock_recovery(processed_signal)
    cpr_signal = carrier_phase_recovery(cr_signal, m_qam=m_qam, window_size=cpr_window)
    decided, evm = symbol_detect(cpr_signal, m_qam=m_qam)
    q_db = compute_q_factor_from_evm(evm)

    return DSPResult(
        compensated=cpr_signal,
        decided_symbols=decided,
        evm=evm,
        q_factor_db=q_db,
    )


def q_factor_baseline(rx_signal: np.ndarray,
                      scene: SceneParams,
                      fs_hz: float,
                      m_qam: int = 16) -> float:
    """
    Compute baseline Q-factor (no NLC, only linear DSP: EDC + CPR).

    This is the baseline against which all NLC algorithms are compared.
    """
    result = process_receiver(rx_signal, scene, fs_hz, m_qam=m_qam)
    return result.q_factor_db
