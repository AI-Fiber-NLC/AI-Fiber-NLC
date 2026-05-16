# AI-Fiber-NLC 🌊

> **Open-source framework for AI-based nonlinear compensation in optical fiber systems**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Tests](https://github.com/AI-Fiber-NLC/AI-Fiber-NLC/actions/workflows/test.yml/badge.svg)](https://github.com/AI-Fiber-NLC/AI-Fiber-NLC/actions/workflows/test.yml)

---

## Vision

Modern coherent optical fiber systems are approaching the physical limits defined by the **nonlinear Shannon limit**. Kerr-effect-induced nonlinear phase noise is the core bottleneck limiting transmission distance and spectral efficiency.

This project builds an open, modular, reproducible AI framework for nonlinear compensation (NLC), bridging academic research and industrial deployment.

**Core thesis:** Upgrading only the terminal equipment (transceivers/DSP) without touching the deployed fiber is the most cost-effective path to higher capacity — enabling operators to leverage the existing 6.5 billion fiber-km already in the ground.

## Quick Start

### Try in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AI-Fiber-NLC/AI-Fiber-NLC/blob/main/notebooks/quickstart_demo.ipynb)

5-minute interactive demo: see constellation diagrams before and after AI compensation.

### Local Install

```bash
git clone https://github.com/AI-Fiber-NLC/AI-Fiber-NLC.git
cd AI-Fiber-NLC
pip install -r requirements.txt

# Or editable install (development mode)
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest tests/ -v
```

### Run Benchmark

```bash
# Validate a benchmark submission
python -m src.benchmark.protocol result.json mvb1
```

## Project Structure

```
AI-Fiber-NLC/
├── src/
│   ├── benchmark/         # Benchmark validation protocol (core)
│   │   └── protocol.py    # Scene definitions + scoring + validation
│   ├── models/            # NLC model implementations
│   │   ├── mlp_nlc.py     # MLP baseline
│   │   ├── baseline_dbp.py# Traditional DBP reference
│   │   └── ...
│   ├── data/              # Fiber simulation data generation
│   └── utils/             # Visualization, config
├── tests/                 # Unit tests
├── benchmark/             # Benchmark configs + results
├── notebooks/             # Interactive demos
└── docs/                  # Documentation
```

## Benchmark Protocol

We use a **result-oriented validation** approach — no subjective review, no complex cryptographic proofs:

| Metric | Description |
|--------|-------------|
| Q-factor improvement (dB) | Raw gain over uncompensated RX |
| Normalized effect (ΔQ_norm) | Score relative to DBP baseline [0, 1.5] |
| Normalized efficiency (E_norm) | Computational cost relative to DBP [0, 1] |
| Composite score | 0.7 × effect + 0.3 × efficiency |

Full protocol: see `src/benchmark/protocol.py` and [docs/benchmark-protocol.md](docs/benchmark-protocol.md).

## Benchmark Scenes

| Scene | Polarization | Modulation | Distance | Key Feature |
|-------|-------------|------------|----------|-------------|
| **MVB-1** | Single-pol | 16QAM | 800 km | Baseline — low compute |
| **MVB-2** | Dual-pol | 16QAM | 800 km | PMD compensation |
| **MVB-3** | Single-pol | 64QAM + PCS | 800 km | Near Shannon limit |

## Contributing

All forms of contribution are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

**Quick path:**
1. Fork the repo
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes
4. Push and open a Pull Request

**Submitting benchmark results?** Use the "Benchmark Submission" issue template and include all required fields.

## License

[MIT License](LICENSE)

## References

- Meta-DSP: [arXiv:2311.10416](https://arxiv.org/abs/2311.10416) (Huawei Noah's Ark Lab)
- FONTE-EID: [GitHub](https://github.com/FONTE-EID/fiber-optic-transmission-system-modeling)
- OptiCommPy: [GitHub](https://github.com/edsonportosilva/OptiCommPy)
- A Survey on ML/DL for Optical Communications: [arXiv:2412.17826](https://arxiv.org/pdf/2412.17826)
