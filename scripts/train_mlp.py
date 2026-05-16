#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""
CLI script to train MLP-NLC models on simulation data.

Usage:
    python scripts/train_mlp.py --scenario mvb1 --power-index 4 --epochs 50
    python scripts/train_mlp.py --scenario mvb1 --model memory --memory-size 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark.protocol import SCENES
from src.models.mlp_nlc import MLP_NLC, MLPWithMemory_NLC


def compute_q_factor(compensated: torch.Tensor, tx: torch.Tensor) -> float:
    """Compute Q-factor from compensated signal."""
    error = compensated - tx
    signal_power = torch.mean(torch.sum(tx ** 2, dim=1))
    error_power = torch.mean(torch.sum(error ** 2, dim=1))
    if error_power > 0 and signal_power > 0:
        evm = torch.sqrt(error_power / signal_power).item()
        q = 20 * np.log10(1.0 / evm) if evm > 0 else 99.9
    else:
        q = 99.9
    return q


def create_windows(rx_flat: torch.Tensor, tx_flat: torch.Tensor,
                   memory_size: int = 0):
    """
    Create windowed samples from flat I/Q tensors.

    Args:
        rx_flat: (N, 2) — received I/Q (must be contiguous in time order)
        tx_flat: (N, 2) — transmitted I/Q
        memory_size: 0 for memoryless, >0 for context window radius

    Returns:
        (rx_out, tx_out):
        - memoryless: (N, 2), (N, 2)
        - windowed:   (N-2m, 2m+1, 2), (N-2m, 2)
    """
    if memory_size == 0:
        return rx_flat, tx_flat

    window = 2 * memory_size + 1
    n = len(rx_flat)
    # Create windows using unfold on contiguous tensor
    rx_windows = rx_flat.unfold(0, window, 1)  # (N-2m, window, 2)
    tx_center = tx_flat[memory_size: n - memory_size]  # (N-2m, 2)
    return rx_windows, tx_center


def train_model(model, train_loader, test_loader, tx_test,
                epochs=50, lr=1e-3, device="cpu"):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "test_loss": [], "test_q": [], "lr": []}
    best_q = -999.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for rx_batch, tx_batch in train_loader:
            rx_batch = rx_batch.to(device)
            tx_batch = tx_batch.to(device)
            optimizer.zero_grad()
            output = model(rx_batch)
            loss = criterion(output, tx_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= n_batches

        model.eval()
        test_loss = 0.0
        n_batches = 0
        all_outputs = []
        with torch.no_grad():
            for rx_batch, tx_batch in test_loader:
                rx_batch = rx_batch.to(device)
                output = model(rx_batch)
                loss = criterion(output, tx_batch.to(device))
                test_loss += loss.item()
                n_batches += 1
                all_outputs.append(output.cpu())
        test_loss /= n_batches
        compensated = torch.cat(all_outputs)
        test_q = compute_q_factor(compensated, tx_test)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        if test_q > best_q:
            best_q = test_q
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        history["train_loss"].append(round(train_loss, 6))
        history["test_loss"].append(round(test_loss, 6))
        history["test_q"].append(round(test_q, 2))
        history["lr"].append(round(current_lr, 8))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"train={train_loss:.6f} | test={test_loss:.6f} | "
                  f"Q={test_q:+.2f} dB | lr={current_lr:.2e}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def main():
    parser = argparse.ArgumentParser(description="Train MLP-NLC model")
    parser.add_argument("--scenario", type=str, default="mvb1", choices=["mvb1", "mvb3"])
    parser.add_argument("--power-index", type=int, default=4)
    parser.add_argument("--all-powers", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model", type=str, default="memory", choices=["simple", "memory"])
    parser.add_argument("--memory-size", type=int, default=5)
    parser.add_argument("--hidden-dims", type=str, default="256,256,128")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--output-dir", type=str, default="./models")
    parser.add_argument("--data-dir", type=str, default="./data/raw")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene = SCENES[args.scenario]
    lo, hi = scene.tx_power_range_dbm
    n_pts = scene.tx_power_points
    powers = np.linspace(lo, hi, n_pts)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Scenario: {scene.name}")
    print(f"Model: {args.model}")
    print(f"Hidden dims: {args.hidden_dims}")
    hidden_dims = [int(d) for d in args.hidden_dims.split(",")]

    memory_size = args.memory_size if args.model == "memory" else 0

    if args.model == "simple":
        model = MLP_NLC(hidden_dims=hidden_dims, dropout=args.dropout)
    else:
        model = MLPWithMemory_NLC(memory_size=memory_size, hidden_dims=hidden_dims,
                                  dropout=args.dropout)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Memory size: {memory_size}")
    print("=" * 60)

    power_indices = range(n_pts) if args.all_powers else [args.power_index]
    all_results = []

    for pidx in power_indices:
        power_dbm = powers[pidx]
        print(f"\n--- Power: {power_dbm:+.1f} dBm (index {pidx}) ---")

        # Load data from .npz
        from src.data.dataset import NLCDataset
        ds = NLCDataset(data_dir, scene.name, power_dbm)
        n = len(ds)

        rx_list, tx_list = [], []
        for i in range(n):
            rx_i, tx_i = ds[i]
            rx_list.append(rx_i)
            tx_list.append(tx_i)
        rx_all = torch.stack(rx_list)  # (N, 2)
        tx_all = torch.stack(tx_list)  # (N, 2)

        # Baseline Q (full dataset)
        q_baseline = compute_q_factor(rx_all, tx_all)
        print(f"  Baseline Q: {q_baseline:+.2f} dB")

        # Train/test split: use contiguous blocks to preserve temporal order
        # for windowed models
        split = int(0.8 * n)
        rx_train_raw, tx_train_raw = rx_all[:split], tx_all[:split]
        rx_test_raw, tx_test_raw = rx_all[split:], tx_all[split:]

        # Create windows (or pass through for memoryless)
        rx_train, tx_train = create_windows(rx_train_raw, tx_train_raw, memory_size)
        rx_test, tx_test = create_windows(rx_test_raw, tx_test_raw, memory_size)

        train_ds = TensorDataset(rx_train, tx_train)
        test_ds = TensorDataset(rx_test, tx_test)

        train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False)

        # Train
        t0 = time.time()
        history = train_model(model, train_loader, test_loader, tx_test,
                              epochs=args.epochs, lr=args.lr, device=device)
        elapsed = time.time() - t0

        best_q = max(history["test_q"])
        best_epoch = history["test_q"].index(best_q) + 1
        print(f"\n  Best Q: {best_q:+.2f} dB (epoch {best_epoch})")
        print(f"  Q improvement: {best_q - q_baseline:+.2f} dB")
        print(f"  Training time: {elapsed:.1f}s")

        # Save
        model_name = f"{args.model}_{scene.name}_p{pidx}"
        model_path = output_dir / f"{model_name}.pt"
        torch.save({
            "model_state": model.state_dict(),
            "model_type": args.model,
            "scene": scene.name,
            "power_dbm": float(power_dbm),
            "hidden_dims": hidden_dims,
            "dropout": args.dropout,
            "memory_size": memory_size,
            "total_params": total_params,
            "history": history,
            "best_q": best_q,
            "q_baseline": q_baseline,
            "training_time_sec": round(elapsed, 1),
        }, model_path)

        hist_path = output_dir / f"{model_name}_history.json"
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)

        all_results.append({
            "power_dbm": float(power_dbm),
            "q_baseline": round(q_baseline, 2),
            "q_best": best_q,
            "improvement": round(best_q - q_baseline, 2),
            "best_epoch": best_epoch,
            "training_time_sec": round(elapsed, 1),
        })

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in all_results:
        print(f"  {r['power_dbm']:+.1f} dBm: "
              f"baseline {r['q_baseline']:+.2f} dB -> "
              f"best {r['q_best']:+.2f} dB "
              f"(+{r['improvement']:+.2f} dB, epoch {r['best_epoch']})")


if __name__ == "__main__":
    main()
