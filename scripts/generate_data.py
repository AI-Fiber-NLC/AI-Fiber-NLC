#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""
CLI script to generate fiber simulation benchmark data.

Usage:
    python scripts/generate_data.py --scenarios mvb1
    python scripts/generate_data.py --scenarios mvb1 mvb2 mvb3
    python scripts/generate_data.py --output-dir ./data --scenarios all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark.protocol import SCENES, SceneParams, scene_to_yaml
from src.data.simulator import FiberSimulator


SCENARIO_MAP = {
    "mvb1": "mvb1",
    "mvb2": "mvb2",
    "mvb3": "mvb3",
    "all": "all",
}


def generate_scenario(
    scene_name: str,
    scene: SceneParams,
    output_dir: Path,
) -> dict:
    """
    Generate data for one scenario.

    Returns:
        Stats dict (duration, file count, total size).
    """
    scene_dir = output_dir / scene.name
    scene_dir.mkdir(parents=True, exist_ok=True)

    # Save scene config as YAML
    yaml_path = scene_dir / "scene.yaml"
    yaml_path.write_text(scene_to_yaml(scene))
    print(f"  Scene config written to {yaml_path}")

    # Create simulator and run power sweep
    sim = FiberSimulator(scene)
    t0 = time.time()
    results = sim.run_power_sweep(progress=False)
    duration = time.time() - t0

    # Save each power point
    file_count = 0
    total_bytes = 0

    for pwr, (rx, tx) in results.items():
        # Ensure consistent dtype for storage
        rx = rx.astype(np.complex64)
        tx = tx.astype(np.complex64)

        fname = f"p_{pwr:.1f}dBm.npz"
        fpath = scene_dir / fname

        np.savez_compressed(
            fpath,
            rx_signal=rx,
            tx_signal=tx,
            tx_power_dbm=np.float64(pwr),
            scene_name=np.str_(scene.name),
            seed=np.int64(scene.seed),
        )

        fsize = fpath.stat().st_size
        total_bytes += fsize
        file_count += 1
        print(f"  {fname}: {fsize / 1024:.1f} KB (rx shape: {rx.shape})")

    return {
        "scene": scene.name,
        "duration_sec": round(duration, 1),
        "file_count": file_count,
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / (1024 * 1024), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AI-Fiber-NLC benchmark data")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/raw",
        help="Output directory (default: ./data/raw)",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["mvb1"],
        choices=["mvb1", "mvb2", "mvb3", "all"],
        help="Scenarios to generate (default: mvb1)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve scenario list
    if "all" in args.scenarios:
        scenario_keys = list(SCENES.keys())
    else:
        scenario_keys = args.scenarios

    print(f"Output directory: {output_dir.resolve()}")
    print(f"Scenarios: {', '.join(scenario_keys)}")
    print("=" * 50)

    all_stats = []
    grand_total_bytes = 0

    for key in scenario_keys:
        scene = SCENES[key]
        print(f"\n{'=' * 40}")
        print(f"Scenario: {scene.name}")
        print(f"  Polarization: {scene.polarization}")
        print(f"  Modulation: {scene.modulation}")
        print(f"  Distance: {scene.fiber_length_km * scene.num_spans:.0f} km")
        print(f"  Symbols: {scene.n_symbols}")
        print(f"  Power range: {scene.tx_power_range_dbm[0]:.1f} to {scene.tx_power_range_dbm[1]:.1f} dBm")
        print(f"  Seed: {scene.seed}")
        print(f"{'=' * 40}")

        stats = generate_scenario(key, scene, output_dir)
        all_stats.append(stats)
        grand_total_bytes += stats["total_bytes"]

        print(f"  Duration: {stats['duration_sec']:.1f}s")
        print(f"  Files: {stats['file_count']}")
        print(f"  Size: {stats['total_mb']:.2f} MB")

    # Summary
    print(f"\n{'=' * 50}")
    print("SUMMARY")
    print(f"{'=' * 50}")
    for s in all_stats:
        print(f"  {s['scene']}: {s['file_count']} files, {s['total_mb']:.2f} MB, {s['duration_sec']:.1f}s")
    print(f"  Total: {grand_total_bytes / (1024*1024):.2f} MB")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
