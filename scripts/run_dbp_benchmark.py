#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""
CLI script to run DBP benchmark on generated simulation data.

Computes Q-factor using the full DSP receiver chain:
- Baseline: EDC → Clock Recovery → CPR → Q-factor
- DBP: DBP → Clock Recovery → CPR → Q-factor

Usage:
    python scripts/run_dbp_benchmark.py --scenario mvb1 --power-index 4
    python scripts/run_dbp_benchmark.py --scenario mvb3 --all-powers
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark.protocol import SCENES, EXTENDED_SCENES, SceneParams
from src.models.baseline_dbp import DBPCompensator
from src.dsp.receiver import process_receiver, DSPResult


def load_data(data_dir: Path, scene_name: str, power_dbm: float):
    """Load rx/tx signal pair from .npz file."""
    from src.data.dataset import NLCDataset
    ds = NLCDataset(data_dir, scene_name, power_dbm)
    _ = ds[0]  # trigger lazy load
    rx = ds._rx.astype(np.complex128)
    tx = ds._tx.astype(np.complex128)
    return rx, tx


def compute_baseline_q(rx: np.ndarray, scene: SceneParams, fs: float, m_qam: int) -> float:
    """Compute baseline Q-factor using DSP chain (EDC + CPR, no NLC)."""
    result = process_receiver(rx, scene, fs, m_qam=m_qam)
    return result.q_factor_db


def compute_dbp_q(rx: np.ndarray, dbp: DBPCompensator,
                  scene: SceneParams, fs: float, m_qam: int,
                  launch_power_dbm: float) -> tuple:
    """Run DBP and compute Q-factor using DSP chain (skip EDC since DBP handles CD)."""
    t0 = time.perf_counter()
    dbp_output = dbp.compensate(rx, launch_power_dbm=launch_power_dbm)
    elapsed = time.perf_counter() - t0

    # After DBP, chromatic dispersion is already compensated,
    # so skip EDC in the DSP chain (use skip_edc=True)
    from src.dsp.receiver import process_receiver
    result = process_receiver(dbp_output, scene, fs, m_qam=m_qam, skip_edc=True)
    return result.q_factor_db, elapsed


def run_single(scenario_key: str, power_index: int, steps_per_span: int,
               data_dir: Path) -> dict:
    """Run DBP benchmark on a single power point."""
    ALL_SCENES = {**SCENES, **EXTENDED_SCENES}
    scene = ALL_SCENES[scenario_key]
    lo, hi = scene.tx_power_range_dbm
    n_pts = scene.tx_power_points
    powers = np.linspace(lo, hi, n_pts)
    power_dbm = powers[power_index]

    fs = scene.baud_rate_GBd * 1e9 * 2  # 2 SPS
    m_qam = 16 if scene.modulation == "16QAM" else 64

    print(f"Scenario: {scene.name}")
    print(f"Power: {power_dbm:+.1f} dBm")
    print(f"Steps per span: {steps_per_span}")
    print(f"Total steps: {scene.num_spans * steps_per_span}")
    print("-" * 40)

    # Load data
    rx, tx = load_data(data_dir, scene.name, power_dbm)
    print(f"Loaded: rx shape={rx.shape}, tx shape={tx.shape}")

    # Baseline (EDC + CPR only)
    print("\nComputing baseline (EDC + CPR)...")
    baseline_q = compute_baseline_q(rx, scene, fs, m_qam)
    print(f"  Baseline Q: {baseline_q:+.2f} dB")

    # DBP
    print(f"\nRunning DBP ({steps_per_span} steps/span)...")
    dbp = DBPCompensator(scene, steps_per_span=steps_per_span)
    dbp_q, dbp_time = compute_dbp_q(rx, dbp, scene, fs, m_qam, power_dbm)
    print(f"  DBP Q: {dbp_q:+.2f} dB")
    print(f"  Improvement: {dbp_q - baseline_q:+.2f} dB")
    print(f"  Processing time: {dbp_time:.2f}s")

    return {
        "scenario": scene.name,
        "power_dbm": float(power_dbm),
        "baseline_q": round(baseline_q, 2),
        "dbp_q": round(dbp_q, 2),
        "improvement": round(dbp_q - baseline_q, 2),
        "dbp_time": round(dbp_time, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Run DBP benchmark")
    parser.add_argument("--scenario", type=str, default="mvb1",
                        choices=["mvb1", "mvb3", "mvb4", "mvb5", "mvb6"])
    parser.add_argument("--power-index", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--data-dir", type=str, default="./data/raw")
    parser.add_argument("--all-powers", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: data directory not found: {data_dir}")
        sys.exit(1)

    # Merge scene registries
    ALL_SCENES = {**SCENES, **EXTENDED_SCENES}
    scene = ALL_SCENES[args.scenario]
    lo, hi = scene.tx_power_range_dbm
    n_pts = scene.tx_power_points
    powers = np.linspace(lo, hi, n_pts)

    if args.all_powers:
        results = []
        for idx in range(n_pts):
            print(f"\n{'=' * 50}")
            print(f"--- Power {idx + 1}/{n_pts}: {powers[idx]:+.1f} dBm ---")
            print(f"{'=' * 50}")
            r = run_single(args.scenario, idx, args.steps, data_dir)
            results.append(r)

        print(f"\n{'=' * 60}")
        print(f"SUMMARY — {scene.name}")
        print(f"{'=' * 60}")
        print(f"{'Power':>8} {'Baseline':>10} {'DBP':>8} {'Improve':>10} {'Time':>8}")
        print("-" * 60)
        for r in results:
            print(f"{r['power_dbm']:+8.1f} {r['baseline_q']:+10.2f} "
                  f"{r['dbp_q']:+8.2f} {r['improvement']:+10.2f} "
                  f"{r['dbp_time']:8.2f}s")
    else:
        run_single(args.scenario, args.power_index, args.steps, data_dir)


if __name__ == "__main__":
    main()
