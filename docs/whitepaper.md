---
title: "AI-Fiber-NLC: An Open-Source Framework for AI-Based Nonlinear Compensation in Optical Fiber Systems"
authors: "AI-Fiber-NLC Contributors"
version: "v0.1.1"
date: "2026-05-17"
license: "MIT"
---

# AI-Fiber-NLC

## An Open-Source Framework for AI-Based Nonlinear Compensation in Optical Fiber Systems

### Version 0.1.1 — White Paper

---

## Abstract

Modern coherent optical fiber systems are rapidly approaching the nonlinear Shannon limit, where Kerr-effect-induced nonlinear phase noise becomes the dominant impairment. Digital Back Propagation (DBP) offers theoretical compensation but suffers from prohibitive computational complexity. Recent advances in deep learning present a new pathway for nonlinear compensation (NLC) with favorable performance-complexity tradeoffs. This paper introduces AI-Fiber-NLC, an open-source, modular benchmark framework for evaluating and developing AI-based NLC algorithms. The framework provides standardized simulation scenes with RRC pulse shaping, a reproducible DSP receiver chain (EDC + clock recovery + carrier phase recovery), composite benchmark scoring, and a contribution system designed to bridge academic research and industrial deployment. We present baseline results from a full DSP receiver chain, DBP, and a memory-augmented MLP model on 16QAM, 800km single-polarization transmission, and document the key finding that at moderate launch powers (+1 to +5 dBm), the bottleneck is EDFA ASE noise rather than nonlinear distortion — defining the operational regime where AI-NLC must demonstrate value.

---

## 1. Introduction

### 1.1 Background

The global fiber-optic communication infrastructure, exceeding 6.5 billion fiber-kilometers, forms the backbone of modern data transmission. As single-channel capacities push beyond 1 Tb/s using probabilistic constellation shaping and higher-order QAM formats, the nonlinear Shannon limit — defined by the interplay of chromatic dispersion and Kerr nonlinearity — has become the primary bottleneck.

Traditional nonlinear compensation techniques, most notably Digital Back Propagation (DBP), can theoretically reverse fiber impairments by numerically solving the inverse nonlinear Schrodinger equation. However, the O(N^2) computational complexity of multi-step DBP makes real-time implementation in commercial DSP chips impractical for long-haul systems.

### 1.2 AI-Driven Compensation

Deep learning has emerged as a promising alternative. Neural networks can learn the inverse mapping of the fiber channel from data, potentially achieving DBP-level compensation at a fraction of the computational cost. Recent work has demonstrated nonlinear phase noise mitigation using convolutional neural networks, recurrent architectures, and meta-learning approaches (e.g., Meta-DSP from Huawei Noah's Ark Lab).

### 1.3 The Gap

Despite growing academic interest, the field lacks a unified, open, reproducible benchmark framework. Results from different research groups are difficult to compare due to varying simulation setups, fiber parameters, and evaluation metrics. This fragmentation slows progress and hinders the translation of research into industrial practice.

### 1.4 Our Contribution

AI-Fiber-NLC addresses this gap by providing:

1. **Standardized benchmark scenes** (MVB series) — well-defined fiber configurations with open parameter specifications, RRC pulse shaping, and 2 samples per symbol
2. **Reproducible DSP receiver chain** — complete signal processing pipeline (EDC + clock recovery + carrier phase recovery) with Q-factor computation
3. **Open-source simulation pipeline** — built on OptiCommPy (MIT licensed), from data generation to model evaluation
4. **Modular model zoo** — a framework for contributing and comparing diverse NLC architectures

---

## 2. Benchmark Protocol

### 2.1 Minimum Viable Benchmark Scenes

We define three standard test scenes:

| Parameter | MVB-1 | MVB-2 | MVB-3 |
|-----------|-------|-------|-------|
| Polarization | Single | Dual | Single |
| Modulation | 16QAM | 16QAM | 64QAM + PCS |
| Baud rate | 32 GBaud | 32 GBaud | 32 GBaud |
| Fiber type | ITU-T G.652 | ITU-T G.652 | ITU-T G.652 |
| Span length | 80 km | 80 km | 80 km |
| Number of spans | 10 | 10 | 10 |
| Total distance | 800 km | 800 km | 800 km |
| Attenuation (α) | 0.2 dB/km | 0.2 dB/km | 0.2 dB/km |
| Dispersion (D) | 16 ps/nm/km | 16 ps/nm/km | 16 ps/nm/km |
| Nonlinearity (γ) | 1.3 1/W/km | 1.3 1/W/km | 1.3 1/W/km |
| PMD | 0 | 0.1 ps/√km | 0 |
| EDFA noise figure | 4.5 dB | 4.5 dB | 4.5 dB |
| Power sweep | -3 to +5 dBm | -3 to +5 dBm | -3 to +5 dBm |
| Symbols | 65,536 | 65,536 | 65,536 |
| Pulse shaping | RRC (α=0.01) | RRC (α=0.01) | RRC (α=0.01) |
| Sampling rate | 2 SPS | 2 SPS | 2 SPS |

MVB-1 serves as the baseline entry point: single-polarization 16QAM at 800km with RRC pulse shaping, capturing essential nonlinear effects without the added complexity of polarization multiplexing.

### 2.2 DSP Receiver Chain

All Q-factor measurements use a standardized DSP receiver chain:

1. **Electronic Dispersion Compensation (EDC)** — frequency-domain equalization removing chromatic dispersion
2. **Clock Recovery** — Gardner algorithm, downsampling from 2 SPS to 1 SPS
3. **Carrier Phase Recovery (CPR)** — Viterbi-Viterbi algorithm (4th power for M-QAM)
4. **Symbol Detection** — nearest-neighbor decision on QAM constellation
5. **Q-factor Computation** — EVM-derived: Q = 20 × log₁₀(1/EVM)

For DBP-compensated signals, EDC is skipped (DBP already compensates chromatic dispersion).

### 2.3 Composite Scoring Formula

Each submitted model is evaluated using a composite score:

```
Score = w_effect × ΔQ_norm + w_efficiency × E_norm
```

**Normalized Q-factor improvement (ΔQ_norm):**
```
ΔQ_norm = (Q_model - Q_baseline) / (Q_dbp - Q_baseline)
```

**Normalized efficiency (E_norm):**
```
E_norm = 1 - log₁₀(FLOPs_model / FLOPs_dbp) / log₁₀(100)
```

**Default weights (Phase 0-1):** w_effect = 0.7, w_efficiency = 0.3

### 2.4 Contribution Verification

Submissions are validated through:
1. Parameter verification (scene, seed, power levels)
2. Reproducibility check (results from submitted model weights)
3. Performance validation (Q-factor on held-out test set)
4. Complexity audit (FLOPs from model architecture)

Cryptographic signing (Ed25519) is in development for tamper-proof contribution attribution.

---

## 3. Baseline Results

### 3.1 Experimental Setup

All results use the MVB-1 scene (16QAM, 800km, single-pol, RRC pulse shaping). Two compensation methods are compared:

1. **DSP Baseline** — EDC + Clock Recovery + Viterbi-Viterbi CPR (no nonlinear compensation)
2. **DBP** — Digital Back Propagation via OptiCommPy SSFM with inverted D and γ, 10 steps/span

### 3.2 Results

| Launch Power | DSP Baseline (dB) | DBP (dB) | DBP Improvement (dB) |
|-------------|-------------------|----------|---------------------|
| -3 dBm | +1.91 | +1.88 | -0.02 |
| -1 dBm | +1.97 | +1.97 | +0.00 |
| 0 dBm | +1.96 | +1.96 | -0.00 |
| +1 dBm | +2.08 | +2.01 | -0.07 |
| +2 dBm | +2.20 | +2.15 | -0.06 |
| +3 dBm | +2.17 | +2.20 | +0.03 |
| +5 dBm | +1.94 | +1.97 | +0.03 |

**Key observations:**

- DBP shows at most +0.03 dB improvement over the DSP baseline at these power levels.
- The Viterbi-Viterbi CPR already compensates for common phase noise (including nonlinear phase rotation), leaving DBP little additional benefit.
- Without CPR, Q-factor drops to +0.16 to +0.29 dB, confirming CPR's dominant role.
- The bottleneck at +1 to +5 dBm is EDFA ASE noise, not nonlinear distortion.

### 3.3 MLP-NLC: DSP Chain Integrated Results

Unlike the preliminary MLP results (Section 3.2), the following experiments
integrate the MLP within the full DSP receiver chain:

```
RX -> EDC -> [MLP-NLC] -> Clock Recovery -> CPR -> Q-factor
```

The MLP is trained on EDC-compensated signals with the TX pulse-shaped signal
as target. After training, the MLP output passes through clock recovery and
CPR before Q-factor computation — ensuring a fair comparison with the baseline.

**Full power sweep results (-3 to +5 dBm, MVB-1):**

| Launch Power | Baseline (dB) | MLP Best (dB) | Improvement | Best Epoch |
|-------------|---------------|---------------|-------------|------------|
| -3.0 dBm | +1.91 | +2.01 | +0.10 | 2 |
| -2.0 dBm | +1.96 | +2.01 | +0.05 | 1 |
| -1.0 dBm | +1.97 | +1.98 | +0.01 | 1 |
| 0.0 dBm | +1.96 | +1.96 | -0.00 | 1 |
| +1.0 dBm | +2.08 | +1.92 | -0.16 | 1 |
| +2.0 dBm | +2.20 | +1.93 | -0.27 | 1 |
| +3.0 dBm | +2.17 | +1.94 | -0.23 | 1 |
| +4.0 dBm | +2.04 | +1.92 | -0.12 | 1 |
| +5.0 dBm | +1.94 | +1.92 | -0.02 | 1 |

**Key observations:**

- **MLP never significantly outperforms the DSP baseline.** At best, +0.10 dB
  improvement at -3 dBm — within measurement noise.
- **MLP overfits immediately.** Best Q-factor always occurs at epoch 1-2.
  Further training causes Q-factor to degrade as the model memorizes ASE noise.
- **At higher powers, MLP makes things worse** (up to -0.27 dB at +2 dBm).
  The MLP fails to learn a useful nonlinear inverse mapping and instead
  corrupts the already well-compensated EDC signal.
- **The performance gap is consistent with Phase 0 findings:** at moderate
  launch powers (+1 to +5 dBm), the bottleneck is EDFA ASE noise, not
  nonlinear distortion. Neither DBP nor MLP can meaningfully improve upon
  what linear DSP (EDC) and phase recovery (CPR) already achieve.

These results define the challenge for AI-NLC: future models must demonstrate
value in regimes where nonlinear effects dominate — either at significantly
higher launch powers (>+7 dBm), longer distances (>1600km), or with
architectures specifically designed for nonlinear phase noise compensation
(Transformer, KAN, or hybrid DSP+AI approaches).

### 3.4 MLP-NLC: Preliminary Results

A memory-augmented MLP (memory_size=5, hidden=[256,256,128], 106K params) was trained on raw RX→TX pairs (MSE loss). Results:

| Method | Q vs Raw Baseline | Notes |
|--------|-------------------|-------|
| Raw baseline (no DSP) | -2.54 dB | Direct RX vs TX comparison |
| MLP-NLC (50 epochs) | +0.25 dB | Improvement: +2.79 dB |
| DSP baseline (EDC+CPR) | +2.08 dB | Meaningful reference |

**Limitation:** The MLP was trained on raw data without the DSP chain. A fair comparison requires training the MLP as a nonlinear compensation block within the DSP chain (RX → EDC → MLP → CPR → Q). This restructuring is planned for Phase 1.

### 3.4 Discussion

These results define the challenge for AI-NLC:

1. **At moderate powers (+1 to +5 dBm)**, nonlinear distortion is weak relative to ASE noise. AI models must demonstrate value beyond what linear DSP (EDC) and phase recovery (CPR) already achieve.
2. **At higher powers (>+5 dBm)**, nonlinear effects dominate and DBP shows more benefit — but these regimes were not tested due to simulation time constraints.
3. **The MLP's raw MSE approach** reduces error relative to uncompensated RX but cannot compete with the DSP chain. Future work should integrate AI models into the DSP chain.

---

## 4. Architecture

### 4.1 Code Organization

```
AI-Fiber-NLC/
├── src/
│   ├── benchmark/          # Protocol: scenes, scoring, validation
│   │   └── protocol.py
│   ├── data/               # Simulation and data loading
│   │   ├── simulator.py    # FiberSimulator (OptiCommPy + RRC)
│   │   └── dataset.py      # NLCDataset (PyTorch Dataset)
│   ├── dsp/                # DSP receiver chain
│   │   └── receiver.py     # EDC + Clock Recovery + CPR + Q
│   ├── models/             # NLC model implementations
│   │   ├── baseline_dbp.py # DBP reference
│   │   └── mlp_nlc.py      # MLP models (simple + memory)
│   ├── training/           # Training utilities
│   └── utils/              # Visualization, config
├── data/
│   └── raw/                # Generated simulation data (RRC, 2 SPS)
│       ├── MVB-1/          # 16QAM, 800km, single-pol
│       └── MVB-3/          # 64QAM+PCS, 800km, single-pol
├── models/                 # Pre-trained model checkpoints
├── scripts/                # CLI tools
│   ├── generate_data.py    # Data generation
│   ├── run_dbp_benchmark.py # DBP evaluation with DSP chain
│   └── train_mlp.py        # MLP training
├── notebooks/              # Interactive demos
│   └── quickstart_demo.ipynb
├── tests/                  # Unit tests (79 tests)
└── docs/                   # Documentation
    └── whitepaper.md       # This document
```

### 4.2 Simulation Pipeline

1. **Symbol generation** — random QAM symbols
2. **RRC pulse shaping** — root-raised cosine filter (α=0.01, span=10 symbols)
3. **Power scaling** — signal scaled to target launch power
4. **Fiber propagation** — OptiCommPy split-step Fourier method, 10 spans with EDFA
5. **Data storage** — compressed .npz format with metadata

### 4.3 Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Simulation | OptiCommPy | MIT license, active development, comprehensive DSP chain |
| Deep learning | PyTorch | Academic standard, flexible, GPU support |
| Benchmarking | Custom DSP chain | Reproducible, transparent, extensible |
| CI/CD | GitHub Actions | Free, integrated, automated testing |

---

## 5. Roadmap

### Phase 0: Foundation — 85% Complete

- [x] Benchmark protocol v0.1
- [x] Fiber simulation with RRC pulse shaping (MVB-1, MVB-3)
- [x] DSP receiver chain (EDC + Clock Recovery + CPR)
- [x] DBP baseline implementation
- [x] MLP-NLC model (preliminary)
- [x] GitHub project infrastructure (CI/CD, templates)
- [x] Colab demo notebook
- [x] Whitepaper v0.1.1
- [ ] MLP-NLC integrated with DSP chain
- [ ] MVB-2 (dual-pol) data — blocked by OptiCommPy manakovSSF bug

### Phase 1: Community Launch (Q3-Q4 2026)

- [ ] MLP-NLC restructured within DSP chain
- [ ] CNN-NLC and Transformer-NLC architectures
- [ ] Comprehensive documentation (ReadTheDocs)
- [ ] Public benchmark leaderboard
- [ ] First benchmark challenge
- [ ] Contributor onboarding guide

### Phase 2: Technical Maturity (Q1-Q3 2027)

- [ ] Multi-scene evaluation (variable distance, submarine fiber)
- [ ] Model Zoo with pre-trained weights (Hugging Face)
- [ ] arXiv paper submission
- [ ] Engagement with optical module vendors
- [ ] Hardware-aware FLOPs estimation

### Phase 3: Sustainability (Q3 2027+)

- [ ] Open-core business model
- [ ] Enterprise features (high-performance inference)
- [ ] SaaS API for NLC-as-a-Service
- [ ] Community governance (RFC process, voting)
- [ ] Core maintainer team

---

## 6. How to Contribute

### Quick Start

1. Fork: https://github.com/AI-Fiber-NLC/AI-Fiber-NLC
2. Install: `pip install -e ".[dev]"`
3. Test: `pytest tests/ -v`
4. Demo: open `notebooks/quickstart_demo.ipynb`

### Contributing Models

1. Read [CONTRIBUTING.md](https://github.com/AI-Fiber-NLC/AI-Fiber-NLC/blob/main/CONTRIBUTING.md)
2. Create model in `src/models/`
3. Train on MVB-1 dataset
4. Submit benchmark results
5. Open Pull Request

---

## 7. References

1. E. Ip and J. M. Kahn, "Compensation of dispersion and nonlinear impairments using digital backpropagation," *JLT*, 26(20), 2008.
2. X. Xiao et al., "Meta-DSP: A Meta-Learning Approach for Data-Driven Nonlinear Compensation," *arXiv:2311.10416*, 2023.
3. E. P. da Silva and A. F. Herbster, "OptiCommPy: Open-source Simulation of Fiber Optic Communications," *JOSS*, 9(98), 2024.
4. G. P. Agrawal, *Nonlinear Fiber Optics*, Elsevier, 2013.

---

*AI-Fiber-NLC v0.1.1 White Paper — May 17, 2026*
*For questions: https://github.com/AI-Fiber-NLC/AI-Fiber-NLC/discussions*
