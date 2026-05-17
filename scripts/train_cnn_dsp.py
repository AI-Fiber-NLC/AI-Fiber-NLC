#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""
Train CNN-NLC integrated within the DSP receiver chain.

DSP chain: RX -> EDC -> [CNN-NLC] -> Clock Recovery -> CPR -> Q-factor

The CNN operates on sequences of EDC-compensated I/Q samples,
leveraging convolutional context to capture dispersion-induced ISI.

Optimized: uses MSE during training, Q-factor computed only at end.

Usage:
    python scripts/train_cnn_dsp.py --scenario mvb1 --power-index 4 --epochs 30
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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark.protocol import SCENES, SceneParams
from src.models.cnn_nlc import CNN_NLC, ResCNN_NLC
from src.dsp.receiver import (
    edc_compensate, clock_recovery, carrier_phase_recovery,
    symbol_detect, compute_q_factor_from_evm
)


def q_factor_through_chain(signal: np.ndarray, scene: SceneParams,
                           fs: float, m_qam: int, skip_edc: bool = False) -> float:
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
    return q_factor_through_chain(rx, scene, fs, m_qam, skip_edc=False)


def create_cnn_training_data(rx: np.ndarray, tx: np.ndarray,
                              scene: SceneParams, fs: float,
                              seq_len: int = 32) -> tuple:
    edc_rx = edc_compensate(rx, scene, fs)
    min_len = min(len(edc_rx), len(tx))
    edc_rx = edc_rx[:min_len]
    tx_target = tx[:min_len]

    edc_iq = np.stack([edc_rx.real, edc_rx.imag], axis=1).astype(np.float32)
    tx_iq = np.stack([tx_target.real, tx_target.imag], axis=1).astype(np.float32)

    n = len(edc_iq) - seq_len + 1
    if n <= 0:
        raise ValueError(f"Signal too short for seq_len={seq_len}")

    # Strided view for efficiency (no copy)
    shape_X = (n, seq_len, 2)
    strides_X = (edc_iq.strides[0], edc_iq.strides[0], edc_iq.strides[1])
    X = np.lib.stride_tricks.as_strided(edc_iq, shape=shape_X, strides=strides_X).copy()

    shape_y = (n, seq_len, 2)
    strides_y = (tx_iq.strides[0], tx_iq.strides[0], tx_iq.strides[1])
    y = np.lib.stride_tricks.as_strided(tx_iq, shape=shape_y, strides=strides_y).copy()

    X = torch.from_numpy(X)
    y = torch.from_numpy(y)

    split = int(0.8 * len(X))
    perm = torch.randperm(len(X))
    train_idx = perm[:split]
    test_idx = perm[split:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def train_model(model: nn.Module, train_loader: DataLoader,
                test_loader: DataLoader, X_test: torch.Tensor,
                y_test: torch.Tensor, scene: SceneParams,
                fs: float, m_qam: int, seq_len: int = 32,
                epochs: int = 50, lr: float = 1e-3,
                device: str = "cpu") -> dict:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "test_loss": [], "test_q": [], "lr": []}
    best_loss = float('inf')
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
            min_len = min(output.shape[1], y_batch.shape[1])
            loss = criterion(output[:, :min_len, :], y_batch[:, :min_len, :])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= n_batches

        # Test MSE (fast)
        model.eval()
        test_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                output = model(X_batch.to(device))
                min_len = min(output.shape[1], y_batch.shape[1])
                loss = criterion(output[:, :min_len, :], y_batch.to(device)[:, :min_len, :])
                test_loss += loss.item()
                n_batches += 1

        test_loss /= n_batches

        if test_loss < best_loss:
            best_loss = test_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Q-factor only every 10 epochs and at end
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            n_q = min(2000, len(X_test))
            sample_outputs = []
            with torch.no_grad():
                for i in range(0, n_q, 256):
                    end_i = min(i + 256, n_q)
                    X_s = X_test[i:end_i].to(device)
                    out = model(X_s)
                    min_len = min(out.shape[1], X_s.shape[1])
                    sample_outputs.append(out.cpu()[:, :min_len, :])
            if sample_outputs:
                compensated_iq = torch.cat(sample_outputs).numpy()
                compensated_flat = (compensated_iq[:, :, 0] + 1j * compensated_iq[:, :, 1]).flatten()
                test_q = q_factor_through_chain(compensated_flat, scene, fs,
                                                 m_qam, skip_edc=True)
            else:
                test_q = 0.0
        else:
            test_q = history["test_q"][-1] if history["test_q"] else 0.0

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

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
    parser = argparse.ArgumentParser(description="Train CNN-NLC in DSP chain")
    parser.add_argument("--scenario", type=str, default="mvb1",
                        choices=["mvb1", "mvb3"])
    parser.add_argument("--power-index", type=int, default=4)
    parser.add_argument("--all-powers", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model", type=str, default="cnn",
                        choices=["cnn", "rescnn"])
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--channels", type=str, default="32,64,64,32")
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
    fs = scene.baud_rate_GBd * 1e9 * 2
    m_qam = 16 if scene.modulation == "16QAM" else 64

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Scenario: {scene.name}")
    print(f"Model: {args.model} (seq_len={args.seq_len})")
    print(f"Channels: {args.channels}")
    channels = [int(c) for c in args.channels.split(",")]

    if args.model == "cnn":
        model = CNN_NLC(in_channels=2, out_channels=2,
                        hidden_channels=channels, kernel_size=7,
                        num_blocks=3, dropout=args.dropout)
    else:
        model = ResCNN_NLC(in_channels=2, out_channels=2,
                           num_layers=8, channels=64,
                           kernel_size=5, dilation_base=2,
                           dropout=args.dropout)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print("=" * 60)

    power_indices = range(n_pts) if args.all_powers else [args.power_index]
    all_results = []

    for pidx in power_indices:
        power_dbm = powers[pidx]
        print(f"\n--- Power: {power_dbm:+.1f} dBm (index {pidx}) ---")

        from src.data.dataset import NLCDataset
        ds = NLCDataset(data_dir, scene.name, power_dbm)
        _ = ds[0]
        rx = ds._rx.astype(np.complex128)
        tx = ds._tx.astype(np.complex128)

        baseline_q = compute_baseline_q(rx, scene, fs, m_qam)
        print(f"  Baseline Q (EDC+CR+CPR): {baseline_q:+.2f} dB")

        print(f"  Creating training data (seq_len={args.seq_len})...")
        try:
            X_train, X_test, y_train, y_test = create_cnn_training_data(
                rx, tx, scene, fs, seq_len=args.seq_len
            )
        except ValueError as e:
            print(f"  ERROR: {e}")
            continue

        print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

        train_ds = TensorDataset(X_train, y_train)
        test_ds = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

        t0 = time.time()
        history = train_model(model, train_loader, test_loader, X_test, y_test,
                              scene, fs, m_qam, args.seq_len,
                              epochs=args.epochs, lr=args.lr, device=device)
        elapsed = time.time() - t0

        # Compute final Q-factor on full test set
        print("  Computing final Q-factor...")
        model.eval()
        all_outputs = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                out = model(X_batch.to(device))
                all_outputs.append(out.cpu())
        compensated_iq = torch.cat(all_outputs).numpy()
        compensated_flat = (compensated_iq[:, :, 0] + 1j * compensated_iq[:, :, 1]).flatten()
        final_q = q_factor_through_chain(compensated_flat, scene, fs,
                                          m_qam, skip_edc=True)

        best_q = max(history["test_q"]) if history["test_q"] else final_q
        best_epoch = history["test_q"].index(best_q) + 1 if best_q in history["test_q"] else 0

        print(f"\n  Final Q (full test): {final_q:+.2f} dB")
        print(f"  Best Q (during training): {best_q:+.2f} dB (epoch {best_epoch})")
        print(f"  vs Baseline: {final_q - baseline_q:+.2f} dB")
        print(f"  Training time: {elapsed:.1f}s")

        model_name = f"dsp_{args.model}_{scene.name}_p{pidx}"
        model_path = output_dir / f"{model_name}.pt"
        torch.save({
            "model_state": model.state_dict(),
            "model_type": args.model,
            "scene": scene.name,
            "power_dbm": float(power_dbm),
            "channels": channels,
            "seq_len": args.seq_len,
            "dropout": args.dropout,
            "total_params": total_params,
            "history": history,
            "best_q": best_q,
            "final_q": final_q,
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
            "final_q": round(final_q, 2),
            "best_q": round(best_q, 2),
            "improvement": round(final_q - baseline_q, 2),
            "training_time_sec": round(elapsed, 1),
        })

    print("\n" + "=" * 60)
    print("SUMMARY — DSP Chain Integrated CNN-NLC")
    print("=" * 60)
    for r in all_results:
        print(f"  {r['power_dbm']:+.1f} dBm: "
              f"baseline {r['baseline_q']:+.2f} dB -> "
              f"CNN {r['final_q']:+.2f} dB "
              f"({r['improvement']:+.2f} dB, {r['training_time_sec']:.0f}s)")


if __name__ == "__main__":
    main()
