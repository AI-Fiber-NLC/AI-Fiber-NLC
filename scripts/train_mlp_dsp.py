#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""
Train MLP-NLC integrated within the DSP receiver chain.

DSP chain: RX -> EDC -> [MLP-NLC] -> Clock Recovery -> CPR -> Q-factor

The MLP operates on the EDC-compensated signal at 2 SPS and outputs
a compensated signal that is then passed through clock recovery and CPR.

Usage:
    python scripts/train_mlp_dsp.py --scenario mvb1 --power-index 4 --epochs 50
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Project root
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark.protocol import SCENES, SceneParams
from src.models.mlp_nlc import MLP_NLC, MLPWithMemory_NLC
from src.dsp.receiver import (
    edc_compensate, clock_recovery, carrier_phase_recovery,
    symbol_detect, compute_q_factor_from_evm, DSPResult
)


def q_factor_through_chain(signal: np.ndarray, scene: SceneParams,
                           fs: float, m_qam: int, skip_edc: bool = False) -> float:
    """
    Compute Q-factor by passing signal through DSP chain.

    If skip_edc=True, skip EDC (for signals already dispersion-compensated).
    """
    if skip_edc:
        processed = signal
    else:
        processed = edc_compensate(signal, scene, fs)

    cr = clock_recovery(processed)
    cpr = carrier_phase_recovery(cr, m_qam=m_qam)
    _, evm = symbol_detect(cpr, m_qam=m_qam)
    return compute_q_factor_from_evm(evm)


def compute_baseline_q(rx: np.ndarray, scene: SceneParams,
                       fs: float, m_qam: int) -> float:
    """Baseline: EDC -> CR -> CPR -> Q (no MLP)."""
    return q_factor_through_chain(rx, scene, fs, m_qam, skip_edc=False)


def create_dsp_training_data(rx: np.ndarray, tx: np.ndarray,
                              scene: SceneParams, fs: float,
                              memory_size: int = 0) -> tuple:
    """
    Create training data with EDC pre-compensation.

    Input to MLP: EDC-compensated signal (with optional context window)
    Target: TX signal (pulse-shaped, at 2 SPS)

    Args:
        rx: received signal at 2 SPS
        tx: transmitted signal at 2 SPS (pulse-shaped)
        scene: benchmark scene
        fs: sampling frequency
        memory_size: context window radius

    Returns:
        (X_train, X_test, y_train, y_test) as torch tensors
    """
    # Apply EDC to RX
    edc_rx = edc_compensate(rx, scene, fs)

    # Trim to same length (EDC may change length slightly)
    min_len = min(len(edc_rx), len(tx))
    edc_rx = edc_rx[:min_len]
    tx_target = tx[:min_len]

    # Convert to real I/Q format
    edc_iq = np.stack([edc_rx.real, edc_rx.imag], axis=1).astype(np.float32)
    tx_iq = np.stack([tx_target.real, tx_target.imag], axis=1).astype(np.float32)

    if memory_size == 0:
        # Memoryless: per-sample
        X = torch.from_numpy(edc_iq)
        y = torch.from_numpy(tx_iq)
    else:
        # With context window
        window = 2 * memory_size + 1
        n = len(edc_iq) - 2 * memory_size
        X = torch.zeros(n, window, 2, dtype=torch.float32)
        y = torch.zeros(n, 2, dtype=torch.float32)
        for i in range(n):
            X[i] = torch.from_numpy(edc_iq[i:i + window])
            y[i] = torch.from_numpy(tx_iq[i + memory_size])

    # Train/test split (80/20)
    split = int(0.8 * len(X))
    perm = torch.randperm(len(X))
    train_idx = perm[:split]
    test_idx = perm[split:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def train_model(model: nn.Module, train_loader: DataLoader,
                test_loader: DataLoader, X_test: torch.Tensor,
                y_test: torch.Tensor, scene: SceneParams,
                fs: float, m_qam: int, memory_size: int = 0,
                epochs: int = 50, lr: float = 1e-3,
                device: str = "cpu") -> dict:
    """
    Train MLP to compensate nonlinear distortion after EDC.

    The loss is MSE between MLP output and TX signal.
    The evaluation metric is Q-factor through the full DSP chain.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "test_loss": [], "test_q": [], "lr": []}
    best_q = -999.0
    best_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= n_batches

        # Evaluation: compute Q-factor through DSP chain
        model.eval()
        test_loss = 0.0
        n_batches = 0
        all_outputs = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                output = model(X_batch)
                loss = criterion(output, y_batch.to(device))
                test_loss += loss.item()
                n_batches += 1
                all_outputs.append(output.cpu())

        test_loss /= n_batches
        compensated_iq = torch.cat(all_outputs).numpy()

        # Reconstruct complex signal from I/Q
        if memory_size > 0:
            # Windowed model: outputs center sample only
            compensated_complex = compensated_iq[:, 0] + 1j * compensated_iq[:, 1]
        else:
            compensated_complex = compensated_iq[:, 0] + 1j * compensated_iq[:, 1]

        # Pass through remaining DSP chain (Clock Recovery + CPR, no EDC)
        test_q = q_factor_through_chain(compensated_complex, scene, fs,
                                         m_qam, skip_edc=True)

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
    parser = argparse.ArgumentParser(description="Train MLP-NLC in DSP chain")
    parser.add_argument("--scenario", type=str, default="mvb1",
                        choices=["mvb1", "mvb3"])
    parser.add_argument("--power-index", type=int, default=4)
    parser.add_argument("--all-powers", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model", type=str, default="memory",
                        choices=["simple", "memory"])
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
    fs = scene.baud_rate_GBd * 1e9 * 2  # 2 SPS
    m_qam = 16 if scene.modulation == "16QAM" else 64

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Scenario: {scene.name}")
    print(f"Model: {args.model} (DSP chain integrated)")
    print(f"Hidden dims: {args.hidden_dims}")
    hidden_dims = [int(d) for d in args.hidden_dims.split(",")]

    memory_size = args.memory_size if args.model == "memory" else 0

    if args.model == "simple":
        model = MLP_NLC(input_dim=2, hidden_dims=hidden_dims,
                        output_dim=2, dropout=args.dropout)
    else:
        model = MLPWithMemory_NLC(memory_size=memory_size,
                                   hidden_dims=hidden_dims,
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

        # Load data
        from src.data.dataset import NLCDataset
        ds = NLCDataset(data_dir, scene.name, power_dbm)
        _ = ds[0]  # trigger lazy load
        rx = ds._rx.astype(np.complex128)
        tx = ds._tx.astype(np.complex128)

        # Baseline Q (EDC + CR + CPR, no MLP)
        baseline_q = compute_baseline_q(rx, scene, fs, m_qam)
        print(f"  Baseline Q (EDC+CR+CPR): {baseline_q:+.2f} dB")

        # Create training data with EDC pre-compensation
        print("  Creating training data (EDC pre-compensated)...")
        X_train, X_test, y_train, y_test = create_dsp_training_data(
            rx, tx, scene, fs, memory_size
        )
        print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

        train_ds = TensorDataset(X_train, y_train)
        test_ds = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False)

        # Train
        t0 = time.time()
        history = train_model(model, train_loader, test_loader, X_test, y_test,
                              scene, fs, m_qam, memory_size,
                              epochs=args.epochs, lr=args.lr, device=device)
        elapsed = time.time() - t0

        best_q = max(history["test_q"])
        best_epoch = history["test_q"].index(best_q) + 1
        print(f"\n  Best Q: {best_q:+.2f} dB (epoch {best_epoch})")
        print(f"  vs Baseline: {best_q - baseline_q:+.2f} dB")
        print(f"  Training time: {elapsed:.1f}s")

        # Save
        model_name = f"dsp_{args.model}_{scene.name}_p{pidx}"
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
            "baseline_q": baseline_q,
            "training_time_sec": round(elapsed, 1),
            "dsp_integrated": True,
        }, model_path)

        hist_path = output_dir / f"{model_name}_history.json"
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)

        all_results.append({
            "power_dbm": float(power_dbm),
            "baseline_q": round(baseline_q, 2),
            "best_q": best_q,
            "improvement": round(best_q - baseline_q, 2),
            "best_epoch": best_epoch,
            "training_time_sec": round(elapsed, 1),
        })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY — DSP Chain Integrated MLP-NLC")
    print("=" * 60)
    for r in all_results:
        print(f"  {r['power_dbm']:+.1f} dBm: "
              f"baseline {r['baseline_q']:+.2f} dB -> "
              f"best {r['best_q']:+.2f} dB "
              f"({r['improvement']:+.2f} dB, epoch {r['best_epoch']})")


if __name__ == "__main__":
    main()
