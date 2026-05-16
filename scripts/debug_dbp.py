"""Debug DBP: check signal properties at each stage."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from src.benchmark.protocol import MVB1
from src.models.baseline_dbp import DBPCompensator

# Load data
data = np.load("data/raw/MVB-1/p_1.0dBm.npz")
rx = data["rx_signal"]
tx = data["tx_signal"]

print(f"RX shape: {rx.shape}, dtype: {rx.dtype}")
print(f"TX shape: {tx.shape}, dtype: {tx.dtype}")
print(f"RX power: {np.mean(np.abs(rx)**2):.4f}")
print(f"TX power: {np.mean(np.abs(tx)**2):.4f}")

# Run DBP with just 1 span, 1 step
dbp_tiny = DBPCompensator(MVB1, steps_per_span=1)
# Manually override to just 1 span
dbp_tiny.n_spans = 1
dbp_tiny.flops_per_symbol = 0

out1 = dbp_tiny.compensate(rx, launch_power_dbm=1.0)
print(f"\nAfter 1 span, 1 step:")
print(f"  Output power: {np.mean(np.abs(out1)**2):.4f}")
print(f"  Output min/max: {np.min(np.abs(out1)):.4f} / {np.max(np.abs(out1)):.4f}")

# Run DBP with 10 spans, 1 step
dbp_10 = DBPCompensator(MVB1, steps_per_span=1)
out10 = dbp_10.compensate(rx, launch_power_dbm=1.0)
print(f"\nAfter 10 spans, 1 step:")
print(f"  Output power: {np.mean(np.abs(out10)**2):.4f}")
print(f"  Output min/max: {np.min(np.abs(out10)):.4f} / {np.max(np.abs(out10)):.4f}")

# Check if signal is exploding
print(f"\n  Any inf? {np.any(np.isinf(out10))}")
print(f"  Any nan? {np.any(np.isnan(out10))}")

# Run DBP with 10 spans, 10 steps
dbp_full = DBPCompensator(MVB1, steps_per_span=10)
out_full = dbp_full.compensate(rx, launch_power_dbm=1.0)
print(f"\nAfter 10 spans, 10 steps:")
print(f"  Output power: {np.mean(np.abs(out_full)**2):.4f}")
print(f"  Output min/max: {np.min(np.abs(out_full)):.4f} / {np.max(np.abs(out_full)):.4f}")
print(f"  Any inf? {np.any(np.isinf(out_full))}")

# Compare with TX
error_1 = np.mean(np.abs(out1 - tx)**2)
error_10 = np.mean(np.abs(out10 - tx)**2)
error_full = np.mean(np.abs(out_full - tx)**2)
error_rx = np.mean(np.abs(rx - tx)**2)

print(f"\nMean squared error vs TX:")
print(f"  RX (no comp): {error_rx:.4f}")
print(f"  DBP 1 span 1 step: {error_1:.4f}")
print(f"  DBP 10 spans 1 step: {error_10:.4f}")
print(f"  DBP 10 spans 10 steps: {error_full:.4f}")
