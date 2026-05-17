#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Generate benchmark result figures for the paper and README."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt

# Results data from benchmarks
results = {
    "MVB-1 (800km, 16QAM)": {
        "powers": [-3, -2, -1, 0, 1, 2, 3, 4, 5],
        "baseline": [1.91, 1.96, 1.97, 1.96, 2.08, 2.20, 2.17, 2.04, 1.94],
        "dbp":      [1.88, 1.97, 1.98, 1.96, 2.01, 2.15, 2.20, 2.04, 1.97],
        "mlp":      [2.01, 2.01, 1.98, 1.96, 1.92, 1.93, 1.94, 1.92, 1.92],
    },
    "MVB-4 (800km, 16QAM, High Power)": {
        "powers": [7, 8, 9, 10],
        "baseline": [2.07, 2.01, 2.04, 2.04],
        "dbp":      [2.09, 2.01, 2.05, 2.03],
    },
    "MVB-5 (1600km, 16QAM, Long Distance)": {
        "powers": [0, 1, 2, 3],
        "baseline": [2.16, 2.04, 1.92, 2.05],
        "dbp":      [2.19, 2.06, 1.92, 2.01],
    },
}

# Create results directory
results_dir = Path(__file__).parent.parent / "results"
results_dir.mkdir(exist_ok=True)

# Figure 1: Combined Q-factor vs Power for all scenes
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (title, data) in enumerate(results.items()):
    ax = axes[idx]
    powers = data["powers"]
    
    ax.plot(powers, data["baseline"], "bo-", linewidth=2, markersize=6, label="Baseline (EDC+CPR)")
    if "dbp" in data:
        ax.plot(powers, data["dbp"], "rs--", linewidth=2, markersize=6, label="DBP (10 steps/span)")
    if "mlp" in data:
        ax.plot(powers, data["mlp"], "g^:", linewidth=2, markersize=6, label="MLP-NLC (DSP-integrated)")
    
    ax.set_xlabel("Launch Power (dBm)", fontsize=12)
    ax.set_ylabel("Q-factor (dB)", fontsize=12)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min(powers) - 0.5, max(powers) + 0.5)

plt.suptitle("AI-NLC Benchmark: Q-factor vs Launch Power\nNo meaningful improvement over EDC+CPR baseline in any regime", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(results_dir / "q_factor_vs_power.png", dpi=200, bbox_inches="tight")
plt.savefig(results_dir / "q_factor_vs_power.pdf", bbox_inches="tight")
plt.close()

# Figure 2: DBP Improvement (zoomed in to show near-zero)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (title, data) in enumerate(results.items()):
    ax = axes[idx]
    powers = data["powers"]
    
    if "dbp" in data:
        improvement = [b - a for a, b in zip(data["baseline"], data["dbp"])]
        ax.bar(powers, improvement, color="steelblue", width=0.4, edgecolor="white")
        ax.axhline(y=0, color="red", linewidth=1, linestyle="--", alpha=0.7)
        ax.axhline(y=0.05, color="orange", linewidth=1, linestyle=":", alpha=0.7, label="Measurement noise floor")
        ax.axhline(y=-0.05, color="orange", linewidth=1, linestyle=":", alpha=0.7)
    
    ax.set_xlabel("Launch Power (dBm)", fontsize=12)
    ax.set_ylabel("DBP Improvement (dB)", fontsize=12)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xlim(min(powers) - 0.5, max(powers) + 0.5)
    ax.set_ylim(-0.35, 0.15)

plt.suptitle("DBP Improvement Over Baseline: Maximum +0.03 dB (within measurement noise)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(results_dir / "dbp_improvement.png", dpi=200, bbox_inches="tight")
plt.savefig(results_dir / "dbp_improvement.pdf", bbox_inches="tight")
plt.close()

print(f"Figures saved to {results_dir}/")
print("  - q_factor_vs_power.png/pdf")
print("  - dbp_improvement.png/pdf")
