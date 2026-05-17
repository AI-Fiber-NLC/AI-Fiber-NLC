#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""
Test: Low-precision DBP vs MLP-NLC complexity trade-off.

Hypothesis: MLP-NLC achieves similar Q-factor to low-precision DBP (1-2 steps/span)
at significantly lower computational cost (FLOPs).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.benchmark.protocol import SCENES
from src.models.baseline_dbp import DBPCompensator
from src.dsp.receiver import process_receiver

SCENE_KEY = "mvb1"
POWER_INDICES = [0, 4, 6, 8]  # -3, +1, +3, +5 dBm
DBP_STEPS = [1, 2, 5, 10]
DATA_DIR = Path("./data/raw")

def run_benchmark():
    scene = SCENES[SCENE_KEY]
    lo, hi = scene.tx_power_range_dbm
    n_pts = scene.tx_power_points
    powers = np.linspace(lo, hi, n_pts)
    fs = scene.baud_rate_GBd * 1e9 * 2
    m_qam = 16

    from src.data.dataset import NLCDataset

    # MLP-NLC pre-computed results (from DSP-integrated training)
    mlp_results = {
        -3.0: 2.01,
         1.0: 1.92,
         3.0: 1.94,
         5.0: 1.92,
    }
    mlp_flops = 106114 * 2  # ~212K FLOPs

    print(f"{'='*80}")
    print(f"Complexity vs Performance: DBP steps vs MLP-NLC")
    print(f"{'='*80}")

    for pidx in POWER_INDICES:
        power_dbm = powers[pidx]
        print(f"\n--- Power: {power_dbm:+.1f} dBm ---")

        ds = NLCDataset(DATA_DIR, scene.name, power_dbm)
        _ = ds[0]
        rx = ds._rx.astype(np.complex128)

        # Baseline
        baseline_result = process_receiver(rx, scene, fs, m_qam=m_qam)
        baseline_q = baseline_result.q_factor_db
        print(f"  Baseline (EDC+CPR): {baseline_q:+.2f} dB")

        # DBP at different step counts
        print(f"  {'Steps':>6} {'Q-factor':>10} {'FLOPs/sym':>12} {'vs Baseline':>12} {'vs MLP':>10}")
        print(f"  {'-'*52}")

        for steps in DBP_STEPS:
            dbp = DBPCompensator(scene, steps_per_span=steps)
            t0 = time.perf_counter()
            dbp_out = dbp.compensate(rx, launch_power_dbm=power_dbm)
            elapsed = time.perf_counter() - t0

            dbp_result = process_receiver(dbp_out, scene, fs, m_qam=m_qam, skip_edc=True)
            dbp_q = dbp_result.q_factor_db
            dbp_flops = dbp.get_flops_per_symbol()

            vs_base = dbp_q - baseline_q
            vs_mlp = dbp_q - mlp_results.get(power_dbm, 0)

            print(f"  {steps:>6} {dbp_q:>+10.2f} {dbp_flops:>12,.0f} {vs_base:>+12.2f} {vs_mlp:>+10.2f}")

        # MLP-NLC
        if power_dbm in mlp_results:
            mlp_q = mlp_results[power_dbm]
            vs_base = mlp_q - baseline_q
            print(f"  {'MLP':>6} {mlp_q:>+10.2f} {mlp_flops:>12,} {vs_base:>+12.2f} {'--':>10}")

if __name__ == "__main__":
    run_benchmark()
