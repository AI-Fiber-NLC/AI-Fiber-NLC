---
title: "AI-Fiber-NLC: An Open-Source Framework for AI-Based Nonlinear Compensation in Optical Fiber Systems"
authors: "AI-Fiber-NLC Contributors"
version: "v0.1"
date: "2026-05-16"
license: "MIT"
---

# AI-Fiber-NLC

## An Open-Source Framework for AI-Based Nonlinear Compensation in Optical Fiber Systems

### Version 0.1 — White Paper

---

## Abstract

Modern coherent optical fiber systems are rapidly approaching the nonlinear Shannon limit, where Kerr-effect-induced nonlinear phase noise becomes the dominant impairment. Digital Back Propagation (DBP) offers theoretical compensation but suffers from prohibitive computational complexity. Recent advances in deep learning present a new pathway for nonlinear compensation (NLC) with favorable performance-complexity tradeoffs. This paper introduces AI-Fiber-NLC, an open-source, modular benchmark framework for evaluating and developing AI-based NLC algorithms. The framework provides standardized simulation scenes, reproducible benchmark scoring, and a contribution system designed to bridge academic research and industrial deployment. We present baseline results from DBP and a memory-augmented MLP model on 16QAM, 800km single-polarization transmission, and outline the project roadmap toward multi-scene evaluation, diverse model architectures, and sustainable open-science governance.

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

1. **Standardized benchmark scenes** (MVB series) — well-defined fiber configurations with open parameter specifications
2. **Reproducible scoring protocol** — composite metric combining Q-factor improvement and computational efficiency
3. **Open-source simulation pipeline** — built on OptiCommPy (MIT licensed), from data generation to model evaluation
4. **Modular model zoo** — a framework for contributing and comparing diverse NLC architectures

---

## 2. Benchmark Protocol

### 2.1 Minimum Viable Benchmark Scenes

We define three standard test scenes, each representing a progressively more challenging transmission scenario:

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
| Symbols per simulation | 65,536 | 65,536 | 65,536 |
| Sampling rate | 2 SPS | 2 SPS | 2 SPS |

MVB-1 serves as the baseline entry point: single-polarization 16QAM at 800km, capturing the essential nonlinear effects without the added complexity of polarization multiplexing or higher-order modulation.

MVB-2 introduces dual-polarization with PMD, representing the format used in most commercial 100G/200G systems.

MVB-3 tests the limits with 64QAM and probabilistic constellation shaping, approaching the nonlinear Shannon limit.

### 2.2 Composite Scoring Formula

Each submitted model is evaluated using a composite score that balances compensation quality and computational cost:

```
Score = w_effect × ΔQ_norm + w_efficiency × E_norm
```

**Normalized Q-factor improvement (ΔQ_norm):**
```
ΔQ_norm = (Q_ai - Q_baseline) / (Q_dbp - Q_baseline)
```
- Q_ai: Q-factor achieved by the AI model
- Q_baseline: Q-factor without any compensation (raw RX signal)
- Q_dbp: Q-factor from the DBP reference (10 steps/span)
- Range: [0, 1.5] — values >1.0 indicate the model exceeds DBP performance

**Normalized efficiency (E_norm):**
```
E_norm = 1 - log₁₀(FLOPs_ai / FLOPs_dbp) / log₁₀(FLOPS_MAX_RATIO)
```
- FLOPs_ai: estimated floating-point operations per symbol
- FLOPs_dbp: FLOPs for the DBP reference
- FLOPS_MAX_RATIO = 100 (models exceeding 100× DBP cost receive zero efficiency credit)
- Range: [0, 1]

**Default weights (Phase 0-1):**
- w_effect = 0.7 (performance priority)
- w_efficiency = 0.3 (cost awareness)

These weights reflect the project's early-stage focus on establishing feasibility. As the framework matures, weights may be adjusted through community governance.

### 2.3 Contribution Verification

Submissions are validated through an automated pipeline:

1. **Parameter verification** — scene configuration, random seed, and power levels must match protocol specifications
2. **Reproducibility check** — results must be reproducible from the submitted model weights and configuration
3. **Performance validation** — Q-factor computed on a held-out test set, compared to baseline and DBP references
4. **Complexity audit** — FLOPs estimated from model architecture and validated against reported values

For Phase 0-1, verification is performed by project maintainers. A cryptographic signing mechanism (Ed25519) is in development to provide tamper-proof contribution attribution, with planned migration to on-chain attestation in Phase 2.

---

## 3. Baseline Results

### 3.1 Experimental Setup

All results reported here use the MVB-1 scene (16QAM, 800km, single-pol) at +1 dBm launch power. The fiber simulation uses OptiCommPy's split-step Fourier method with EDFA amplification at each 80km span.

Two compensation methods are compared:

1. **DBP (Digital Back Propagation)** — classical reference, implemented via OptiCommPy's SSFM with inverted dispersion and nonlinearity coefficients, 10 steps per span (100 total steps)
2. **MLP-NLC** — a feedforward neural network with temporal context window (memory_size=5, hidden dims=[256,256,128], 106K parameters), trained for 50 epochs on 80% of the data

### 3.2 Results

| Method | Q-factor (dB) | Improvement (dB) | Processing Time |
|--------|-------------|-----------------|-----------------|
| Baseline (no compensation) | -2.81 | — | — |
| DBP (10 steps/span) | +0.34 | +3.15 | 1.67s |
| MLP-NLC (memory=5, 50 epochs) | +0.06 | +2.87 | 36.7s (train) |

**Key observations:**

- DBP provides the strongest baseline at +3.15 dB improvement, consistent with theoretical expectations for 800km 16QAM transmission.
- The MLP-NLC model achieves +2.87 dB improvement, closing to within 0.28 dB of DBP. With further training and architectural refinement, the gap may be reduced.
- The MLP approach has significantly higher training cost (36.7s) but much lower inference cost per symbol than DBP.
- All Q-factor values are computed as relative metrics (EVM-derived), comparing compensated signals to the known transmitted constellation. Absolute Q-factors depend on receiver DSP components (timing recovery, carrier phase estimation) not yet implemented.

### 3.3 Discussion

The modest gap between MLP-NLC and DBP reflects the current state of the model. Several factors limit performance:

1. **Limited context window** — memory_size=5 captures only local dispersion effects. Chromatic dispersion over 800km spreads energy across hundreds of symbols, requiring larger windows or more sophisticated temporal models.
2. **Training data size** — 65,536 symbols per power point provides limited diversity. Data augmentation and multi-power training may improve generalization.
3. **Model capacity** — 106K parameters is modest for a task of this complexity. Larger models with regularization may achieve better results.
4. **No receiver DSP** — timing recovery, carrier phase estimation, and forward error correction are not included, which limits the absolute Q-factor values.

These limitations define clear improvement paths for Phase 1 and beyond.

---

## 4. Architecture

### 4.1 Code Organization

```
AI-Fiber-NLC/
├── src/
│   ├── benchmark/          # Protocol: scenes, scoring, validation
│   │   └── protocol.py
│   ├── data/               # Simulation and data loading
│   │   ├── simulator.py    # FiberSimulator (OptiCommPy wrapper)
│   │   └── dataset.py      # NLCDataset (PyTorch Dataset)
│   ├── models/             # NLC model implementations
│   │   ├── baseline_dbp.py # DBP reference
│   │   └── mlp_nlc.py      # MLP models
│   ├── training/           # Training utilities
│   └── utils/              # Visualization, config
├── data/
│   └── raw/                # Generated simulation data
│       ├── MVB-1/          # 16QAM, 800km, single-pol
│       └── MVB-3/          # 64QAM+PCS, 800km, single-pol
├── models/                 # Pre-trained model checkpoints
├── scripts/                # CLI tools
│   ├── generate_data.py    # Data generation
│   ├── run_dbp_benchmark.py # DBP evaluation
│   └── train_mlp.py        # MLP training
├── notebooks/              # Interactive demos
│   └── quickstart_demo.ipynb
├── tests/                  # Unit tests
└── docs/                   # Documentation
    └── whitepaper.md       # This document
```

### 4.2 Simulation Pipeline

The data generation pipeline uses OptiCommPy, an MIT-licensed Python framework for optical communication simulation:

1. **Symbol generation** — random QAM symbols at specified baud rate
2. **Upsampling** — 2 samples per symbol for proper SSFM resolution
3. **Power scaling** — signal scaled to target launch power
4. **Fiber propagation** — split-step Fourier method through 10 spans with EDFA amplification
5. **Data storage** — compressed .npz format with metadata

### 4.3 Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Simulation | OptiCommPy | MIT license, active development, comprehensive DSP chain |
| Deep learning | PyTorch | Academic standard, flexible, GPU support |
| Benchmarking | Custom protocol | Transparent, reproducible, extensible |
| CI/CD | GitHub Actions | Free, integrated, automated testing |
| Documentation | ReadTheDocs | Standard for open-source projects |

---

## 5. Roadmap

### Phase 0: Foundation (Current) — Complete

- [x] Benchmark protocol v0.1
- [x] Fiber simulation data (MVB-1, MVB-3)
- [x] DBP baseline implementation
- [x] MLP-NLC model and training pipeline
- [x] GitHub project infrastructure (CI/CD, templates, issues)
- [x] Colab demo notebook

### Phase 1: Community Launch (Q3-Q4 2026)

- [ ] Additional model architectures (CNN-NLC, Transformer-NLC, KAN-NLC)
- [ ] MVB-2 (dual-polarization) support
- [ ] Comprehensive documentation (ReadTheDocs)
- [ ] Public benchmark leaderboard
- [ ] First benchmark challenge/competition
- [ ] Contributor onboarding guide

### Phase 2: Technical Maturity (Q1-Q3 2027)

- [ ] Multi-scene evaluation (variable distance, submarine fiber)
- [ ] Model Zoo with pre-trained weights (Hugging Face)
- [ ] arXiv paper submission
- [ ] Engagement with optical module vendors
- [ ] Hardware-aware FLOPs estimation (FPGA/ASIC)

### Phase 3: Sustainability (Q3 2027+)

- [ ] Open-core business model definition
- [ ] Enterprise features (high-performance inference engine)
- [ ] SaaS API for NLC-as-a-Service
- [ ] Community governance (RFC process, voting)
- [ ] Core maintainer team establishment

---

## 6. How to Contribute

We welcome contributions in all forms: new model architectures, improved simulation methods, documentation, benchmark data, and community engagement.

### Quick Start

1. Fork the repository: https://github.com/AI-Fiber-NLC/AI-Fiber-NLC
2. Install dependencies: `pip install -e ".[dev]"`
3. Run tests: `pytest tests/ -v`
4. Try the demo: open `notebooks/quickstart_demo.ipynb`

### Contributing Models

1. Read the [CONTRIBUTING.md](https://github.com/AI-Fiber-NLC/AI-Fiber-NLC/blob/main/CONTRIBUTING.md) guide
2. Create a new model in `src/models/` following the `BaseNLC` interface pattern
3. Train on the standard MVB-1 dataset
4. Submit benchmark results using the Benchmark Submission template
5. Open a Pull Request

### Technical Discussions

Use GitHub Discussions for architecture proposals, benchmark methodology questions, and general project feedback.

---

## 7. References

1. E. Ip and J. M. Kahn, "Compensation of dispersion and nonlinear impairments using digital backpropagation," *Journal of Lightwave Technology*, 26(20), 2008.
2. X. Xiao et al., "Meta-DSP: A Meta-Learning Approach for Data-Driven Nonlinear Compensation in High-Speed Optical Fiber Systems," *arXiv:2311.10416*, 2023.
3. E. P. da Silva and A. F. Herbster, "OptiCommPy: Open-source Simulation of Fiber Optic Communications with Python," *JOSS*, 9(98), 2024.
4. G. P. Agrawal, *Nonlinear Fiber Optics*, Elsevier Science, 2013.
5. A Survey on Machine and Deep Learning for Optical Communications, *arXiv:2412.17826*, 2024.

---

## 8. License

This project is licensed under the MIT License. See [LICENSE](https://github.com/AI-Fiber-NLC/AI-Fiber-NLC/blob/main/LICENSE) for details.

All simulation data is generated using OptiCommPy (MIT License) and is provided under the same license.

---

*AI-Fiber-NLC v0.1 White Paper — May 16, 2026*
*For questions: https://github.com/AI-Fiber-NLC/AI-Fiber-NLC/discussions*
