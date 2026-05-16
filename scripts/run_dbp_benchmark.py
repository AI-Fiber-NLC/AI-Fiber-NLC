#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""
CLI script to run DBP benchmark on generated simulation data.

Usage:
    python scripts/run_dbp_benchmark.py --scenario mvb1 --power-index 4
    python scripts/run_dbp_benchmark.py --scenario mvb3 --power-index 0 --steps 20
    python scripts/run_dbp_benchmark.py --scenario mvb1 --all-powers
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark.protocol import SCENES, SceneParams
from src.models.baseline_dbp import DBPCompensator


def load_data(data_dir: Path, scene_name: str, power_dbm: float):
    """Load rx/tx signal pair from .npz file."""
    fname = f"p_{power_dbm:.1f}dBm.npz"
    fpath = data_dir / scene_name / fname

    if not fpath.exists():
        print(f"Error: file not found: {fpath}")
        print(f"Run 'python scripts/generate_data.py --scenarios {scene_name.lower()}' first.")
        sys.exit(1)

    data = np.load(fpath)
    rx = data["rx_signal"]
    tx = data["tx_signal"]
    return rx, tx


def run_single(scenario_key: str,
               power_index: int,
               steps_per_span: int,
               data_dir: Path) -> dict:
    """Run DBP on a single power point."""
    scene = SCENES[scenario_key]

    # Power sweep: linspace from scene config
    lo, hi = scene.tx_power_range_dbm
    n_pts = scene.tx_power_points
    powers = np.linspace(lo, hi, n_pts)
    power_dbm = powers[power_index]

    print(f"Scenario: {scene.name}")
    print(f"Power index: {power_index} → {power_dbm:+.1f} dBm")
    print(f"Steps per span: {steps_per_span}")
    print(f"Total steps: {scene.num_spans * steps_per_span}")
    print("-" * 40)

    # Load data
    rx, tx = load_data(data_dir, scene.name, power_dbm)
    print(f"Loaded data: rx shape={rx.shape}, tx shape={tx.shape}")

    # Create compensator
    dbp = DBPCompensator(scene, steps_per_span=steps_per_span)

    # Baseline (no compensation)
    baseline_error = np.mean(np.abs(rx - tx) ** 2)
    signal_power = np.mean(np.abs(tx) ** 2)
    baseline_evm = np.sqrt(baseline_error / signal_power)
    baseline_q = 20 * np.log10(1.0 / baseline_evm) if baseline_evm > 0 else 99.9

    print(f"\nBaseline (no compensation):")
    print(f"  Q-factor: {baseline_q:.2f} dB")
    print(f"  EVM: {baseline_evm:.4f}")

    # Run DBP
    t0 = time.perf_counter()
    compensated = dbp.compensate(rx, launch_power_dbm=power_dbm)
    elapsed = time.perf_counter() - t0

    # DBP metrics
    dbp_error = np.mean(np.abs(compensated - tx) ** 2)
    dbp_evm = np.sqrt(dbp_error / signal_power) if signal_power > 0 else 0
    dbp_q = 20 * np.log10(1.0 / dbp_evm) if dbp_evm > 0 else 99.9

    print(f"\nDBP compensation ({steps_per_span} steps/span):")
    print(f"  Q-factor: {dbp_q:.2f} dB")
    print(f"  EVM: {dbp_evm:.4f}")
    print(f"  Q-factor improvement: {dbp_q - baseline_q:+.2f} dB")
    print(f"  Processing time: {elapsed:.2f}s")
    print(f"  Estimated FLOPs/symbol: {dbp.flops_per_symbol:,.0f}")

    return {
        "scenario": scene.name,
        "power_dbm": float(power_dbm),
        "steps_per_span": steps_per_span,
        "baseline_q": round(baseline_q, 2),
        "dbp_q": round(dbp_q, 2),
        "q_improvement": round(dbp_q - baseline_q, 2),
        "baseline_evm": round(baseline_evm, 4),
        "dbp_evm": round(dbp_evm, 4),
        "processing_time_sec": round(elapsed, 2),
        "flops_per_symbol": dbp.flops_per_symbol,
    }


def run_all_powers(scenario_key: str,
                   steps_per_span: int,
                   data_dir: Path) -> list:
    """Run DBP on all power points for a scenario."""
    scene = SCENES[scenario_key]
    lo, hi = scene.tx_power_range_dbm
    n_pts = scene.tx_power_points

    results = []
    print(f"Running DBP benchmark for {scene.name}")
    print(f"Steps per span: {steps_per_span}")
    print(f"Powers: {n_pts} points from {lo:.1f} to {hi:.1f} dBm")
    print("=" * 50)

    for idx in range(n_pts):
        print(f"\n--- Power point {idx + 1}/{n_pts} ---")
        r = run_single(scenario_key, idx, steps_per_span, data_dir)
        results.append(r)
        print()

    # Summary table
    print("\n" + "=" * 60)
    print(f"{'Power (dBm)':>12} {'Baseline Q':>12} {'DBP Q':>10} {'Improvement':>12} {'Time (s)':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['power_dbm']:+12.1f} {r['baseline_q']:12.2f} {r['dbp_q']:10.2f} "
              f"{r['q_improvement']:+12.2f} {r['processing_time_sec']:10.2f}")
    print("=" * 60)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DBP benchmark on simulation data")
    parser.add_argument(
        "--scenario",
        type=str,
        default="mvb1",
        choices=["mvb1", "mvb3"],
        help="Scenario to benchmark (default: mvb1)",
    )
    parser.add_argument(
        "--power-index",
        type=int,
        default=4,
        help="Power point index (0-8, default: 4 = 1 dBm)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Steps per span (default: 10)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/raw",
        help="Data directory (default: ./data/raw)",
    )
    parser.add_argument(
        "--all-powers",
        action="store_true",
        help="Run all power points instead of a single one",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: data directory not found: {data_dir}")
        print("Run 'python scripts/generate_data.py --scenarios all' first.")
        sys.exit(1)

    if args.all_powers:
        run_all_powers(args.scenario, args.steps, data_dir)
    else:
        run_single(args.scenario, args.power_index, args.steps, data_dir)


if __name__ == "__main__":
    main()
